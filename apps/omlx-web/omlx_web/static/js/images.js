/* Image playground: text-to-image and image editing against /v1/images/*.
 *
 * - Models and their capabilities come from GET /v1/images/models.
 * - Generation uses JSON (POST /v1/images/generations); editing uploads the
 *   references (and an optional painted/uploaded mask) as multipart
 *   (POST /v1/images/edits).
 * - Progress is polled from the admin activity feed (/admin/api/activity),
 *   which carries the image engine's per-step counter.
 * - Results live in IndexedDB (blobs + parameters) so the gallery survives
 *   reloads without touching the server.
 */
(function () {
    'use strict';

    const API_KEY_STORAGE_KEY = 'omlx_chat_api_key';
    const MODEL_STORAGE_KEY = 'omlx_images_model';
    const FORM_STORAGE_KEY = 'omlx_images_form';
    const DB_NAME = 'omlx-images';
    const DB_STORE = 'items';
    const ASPECTS = ['1:1', '4:3', '3:4', '3:2', '2:3', '16:9', '9:16'];
    const RESOLUTIONS = [512, 768, 1024, 1536, 2048];

    const t = (key, vars) => {
        let s = window.t ? window.t(key) : key;
        if (vars) for (const [k, v] of Object.entries(vars)) s = s.replace(`{${k}}`, v);
        return s;
    };

    // ---- IndexedDB ------------------------------------------------------
    function openDb() {
        return new Promise((resolve, reject) => {
            const req = indexedDB.open(DB_NAME, 1);
            req.onupgradeneeded = () => {
                const db = req.result;
                if (!db.objectStoreNames.contains(DB_STORE)) {
                    db.createObjectStore(DB_STORE, { keyPath: 'id' });
                }
            };
            req.onsuccess = () => resolve(req.result);
            req.onerror = () => reject(req.error);
        });
    }

    async function dbTx(mode, fn) {
        const db = await openDb();
        return new Promise((resolve, reject) => {
            const tx = db.transaction(DB_STORE, mode);
            const store = tx.objectStore(DB_STORE);
            const result = fn(store);
            tx.oncomplete = () => resolve(result && result.result !== undefined ? result.result : result);
            tx.onerror = () => reject(tx.error);
        });
    }

    function b64ToBlob(b64, mime) {
        const bin = atob(b64);
        const bytes = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
        return new Blob([bytes], { type: mime });
    }

    function loadImageSize(url) {
        return new Promise((resolve, reject) => {
            const img = new Image();
            img.onload = () => resolve({ w: img.naturalWidth, h: img.naturalHeight, img });
            img.onerror = reject;
            img.src = url;
        });
    }

    window.imagePlayground = function (serverApiKey) {
        return {
            // auth / models
            apiKey: '',
            apiKeyInput: '',
            models: [],
            modelsLoading: true,
            modelId: '',
            // form
            mode: 'generate',
            prompt: '',
            negativePrompt: '',
            aspect: '1:1',
            resolution: 1024,
            customSize: false,
            width: 1024,
            height: 1024,
            followReference: true,
            steps: null,
            guidance: null,
            seed: '',
            lockSeed: false,
            n: 1,
            outputFormat: 'png',
            stepCache: false,
            strength: 1.0,
            showAdvanced: false,
            // edit inputs
            references: [], // {id, blob, url, w, h, name}
            mask: null, // {blob, url}
            dragOver: false,
            // mask painter
            maskEditorOpen: false,
            brushSize: 48,
            eraseMode: false,
            _painting: false,
            // run state
            running: false,
            progress: null, // {done,total,phase}
            elapsed: 0,
            error: '',
            _abort: null,
            _timer: null,
            _poll: null,
            // gallery
            items: [],
            urls: {},
            current: null,
            lightbox: null,
            ASPECTS,
            RESOLUTIONS,

            async init() {
                if (serverApiKey) localStorage.setItem(API_KEY_STORAGE_KEY, serverApiKey);
                this.apiKey = localStorage.getItem(API_KEY_STORAGE_KEY) || '';
                this.restoreForm();
                await this.loadGallery();
                if (this.apiKey) await this.loadModels();
                window.addEventListener('paste', (e) => this.onPaste(e));
                this.$watch('modelId', () => this.onModelChange());
                ['prompt', 'negativePrompt', 'aspect', 'resolution', 'customSize', 'width', 'height',
                 'n', 'outputFormat', 'mode'].forEach((k) => this.$watch(k, () => this.saveForm()));
                // Icons inside x-for / x-if content are rendered after init.
                ['items', 'references', 'current', 'lightbox', 'mode', 'running']
                    .forEach((k) => this.$watch(k, () => this.$nextTick(() => window.lucide && window.lucide.createIcons())));
                this.$nextTick(() => window.lucide && window.lucide.createIcons());
            },

            // ---- persistence ---------------------------------------------
            saveForm() {
                const keep = {
                    prompt: this.prompt, negativePrompt: this.negativePrompt, aspect: this.aspect,
                    resolution: this.resolution, customSize: this.customSize, width: this.width,
                    height: this.height, n: this.n, outputFormat: this.outputFormat, mode: this.mode,
                };
                localStorage.setItem(FORM_STORAGE_KEY, JSON.stringify(keep));
            },
            restoreForm() {
                try {
                    Object.assign(this, JSON.parse(localStorage.getItem(FORM_STORAGE_KEY) || '{}'));
                } catch (e) { /* ignore */ }
            },

            saveApiKey() {
                const key = this.apiKeyInput.trim();
                if (!key) return;
                localStorage.setItem(API_KEY_STORAGE_KEY, key);
                this.apiKey = key;
                this.loadModels();
            },

            headers(extra) {
                return Object.assign({ Authorization: `Bearer ${this.apiKey}` }, extra || {});
            },

            // ---- models ---------------------------------------------------
            async loadModels() {
                this.modelsLoading = true;
                try {
                    const resp = await fetch('/v1/images/models', { headers: this.headers() });
                    if (resp.status === 401) {
                        localStorage.removeItem(API_KEY_STORAGE_KEY);
                        this.apiKey = '';
                        return;
                    }
                    const data = await resp.json();
                    this.models = data.data || [];
                    const saved = localStorage.getItem(MODEL_STORAGE_KEY);
                    const pick = this.models.find((m) => m.id === saved) || this.models[0];
                    this.modelId = pick ? pick.id : '';
                    this.onModelChange();
                } catch (e) {
                    this.error = t('images.error.models');
                } finally {
                    this.modelsLoading = false;
                    this.$nextTick(() => window.lucide && window.lucide.createIcons());
                }
            },

            get model() {
                return this.models.find((m) => m.id === this.modelId) || null;
            },
            get caps() {
                return (this.model && this.model.capabilities) || {};
            },
            get defaults() {
                return (this.model && this.model.defaults) || {};
            },
            get grid() {
                return this.defaults.size_multiple || 16;
            },
            get canEdit() {
                return !!this.caps.edit;
            },
            get maxRefs() {
                return this.caps.max_references || 0;
            },

            onModelChange() {
                if (!this.model) return;
                localStorage.setItem(MODEL_STORAGE_KEY, this.modelId);
                if (!this.canEdit && this.mode === 'edit') this.mode = 'generate';
                if (!this.caps.mask) this.clearMask();
                if (!this.caps.strength) this.strength = 1.0;
                if (!(this.caps.lossy_accel || []).includes('step_cache')) this.stepCache = false;
                if (this.references.length > this.maxRefs && this.maxRefs > 0) {
                    this.references.splice(this.maxRefs);
                }
                this.steps = null;
                this.guidance = null;
                this.recomputeSize();
            },

            // ---- size -----------------------------------------------------
            snap(v) {
                const g = this.grid;
                const min = this.defaults.min_side || 256;
                const max = this.defaults.max_side || 2048;
                return Math.max(min, Math.min(max, Math.round(v / g) * g));
            },
            recomputeSize() {
                if (this.customSize) {
                    this.width = this.snap(this.width);
                    this.height = this.snap(this.height);
                    return;
                }
                const [a, b] = this.aspect.split(':').map(Number);
                const area = this.resolution * this.resolution;
                const w = Math.sqrt((area * a) / b);
                this.width = this.snap(w);
                this.height = this.snap((w * b) / a);
            },
            setAspect(a) {
                this.aspect = a;
                this.customSize = false;
                this.recomputeSize();
            },
            get sizeLabel() {
                if (this.mode === 'edit' && this.followReference) return t('images.size.follow');
                return `${this.width} × ${this.height}`;
            },

            // ---- references ----------------------------------------------
            async addFiles(fileList) {
                const files = Array.from(fileList || []).filter((f) => f.type.startsWith('image/'));
                for (const f of files) {
                    if (this.maxRefs && this.references.length >= this.maxRefs) {
                        this.error = t('images.error.max_refs', { n: this.maxRefs });
                        break;
                    }
                    await this.addReferenceBlob(f, f.name);
                }
                if (files.length && this.canEdit) this.mode = 'edit';
            },
            async addReferenceBlob(blob, name) {
                const url = URL.createObjectURL(blob);
                try {
                    const { w, h } = await loadImageSize(url);
                    this.references.push({ id: crypto.randomUUID(), blob, url, w, h, name: name || 'image' });
                } catch (e) {
                    URL.revokeObjectURL(url);
                    this.error = t('images.error.bad_image');
                }
                this.$nextTick(() => window.lucide && window.lucide.createIcons());
            },
            removeReference(i) {
                const [r] = this.references.splice(i, 1);
                if (r) URL.revokeObjectURL(r.url);
                if (i === 0) this.clearMask();
            },
            onDrop(e) {
                this.dragOver = false;
                this.addFiles(e.dataTransfer.files);
            },
            onPaste(e) {
                if (!this.canEdit) return;
                const files = Array.from(e.clipboardData?.items || [])
                    .filter((it) => it.type.startsWith('image/'))
                    .map((it) => it.getAsFile())
                    .filter(Boolean);
                if (files.length) this.addFiles(files);
            },

            // ---- mask -----------------------------------------------------
            clearMask() {
                if (this.mask) URL.revokeObjectURL(this.mask.url);
                this.mask = null;
            },
            async uploadMask(fileList) {
                const f = fileList && fileList[0];
                if (!f) return;
                this.clearMask();
                this.mask = { blob: f, url: URL.createObjectURL(f) };
            },
            async openMaskEditor() {
                if (!this.references.length) return;
                this.maskEditorOpen = true;
                await this.$nextTick();
                const ref = this.references[0];
                const canvas = this.$refs.maskCanvas;
                const bg = this.$refs.maskBg;
                const maxW = Math.min(window.innerWidth * 0.8, 900);
                const maxH = window.innerHeight * 0.65;
                const scale = Math.min(maxW / ref.w, maxH / ref.h, 1);
                const cw = Math.round(ref.w * scale);
                const ch = Math.round(ref.h * scale);
                bg.style.width = canvas.style.width = `${cw}px`;
                bg.style.height = canvas.style.height = `${ch}px`;
                canvas.width = ref.w;
                canvas.height = ref.h;
                this._maskScale = ref.w / cw;
                const ctx = canvas.getContext('2d');
                ctx.clearRect(0, 0, ref.w, ref.h);
                if (this.mask) {
                    // Show the existing mask as a red overlay.
                    const { img } = await loadImageSize(this.mask.url);
                    const tmp = document.createElement('canvas');
                    tmp.width = ref.w;
                    tmp.height = ref.h;
                    const tctx = tmp.getContext('2d');
                    tctx.drawImage(img, 0, 0, ref.w, ref.h);
                    const data = tctx.getImageData(0, 0, ref.w, ref.h);
                    for (let i = 0; i < data.data.length; i += 4) {
                        const on = data.data[i] > 127;
                        data.data[i] = 255; data.data[i + 1] = 40; data.data[i + 2] = 40;
                        data.data[i + 3] = on ? 140 : 0;
                    }
                    ctx.putImageData(data, 0, 0);
                }
                window.lucide && window.lucide.createIcons();
            },
            maskPoint(e) {
                const rect = this.$refs.maskCanvas.getBoundingClientRect();
                return {
                    x: (e.clientX - rect.left) * this._maskScale,
                    y: (e.clientY - rect.top) * this._maskScale,
                };
            },
            maskDown(e) {
                this._painting = true;
                this._last = this.maskPoint(e);
                this.maskStroke(e);
            },
            maskMove(e) {
                if (this._painting) this.maskStroke(e);
            },
            maskUp() {
                this._painting = false;
            },
            maskStroke(e) {
                const ctx = this.$refs.maskCanvas.getContext('2d');
                const p = this.maskPoint(e);
                ctx.globalCompositeOperation = this.eraseMode ? 'destination-out' : 'source-over';
                ctx.strokeStyle = 'rgba(255,40,40,0.55)';
                ctx.lineWidth = this.brushSize * this._maskScale;
                ctx.lineCap = 'round';
                ctx.lineJoin = 'round';
                ctx.beginPath();
                ctx.moveTo(this._last.x, this._last.y);
                ctx.lineTo(p.x, p.y);
                ctx.stroke();
                this._last = p;
            },
            clearMaskCanvas() {
                const c = this.$refs.maskCanvas;
                c.getContext('2d').clearRect(0, 0, c.width, c.height);
            },
            async saveMaskEditor() {
                const src = this.$refs.maskCanvas;
                const out = document.createElement('canvas');
                out.width = src.width;
                out.height = src.height;
                const octx = out.getContext('2d');
                const data = src.getContext('2d').getImageData(0, 0, src.width, src.height);
                const res = octx.createImageData(src.width, src.height);
                let painted = 0;
                for (let i = 0; i < data.data.length; i += 4) {
                    const v = data.data[i + 3] > 10 ? 255 : 0;
                    if (v) painted++;
                    res.data[i] = res.data[i + 1] = res.data[i + 2] = v;
                    res.data[i + 3] = 255;
                }
                octx.putImageData(res, 0, 0);
                this.maskEditorOpen = false;
                this.clearMask();
                if (!painted) return;
                const blob = await new Promise((r) => out.toBlob(r, 'image/png'));
                this.mask = { blob, url: URL.createObjectURL(blob) };
            },

            // ---- run ------------------------------------------------------
            get canSubmit() {
                if (!this.model || this.running || !this.prompt.trim()) return false;
                if (this.mode === 'edit' && !this.references.length) return false;
                return true;
            },

            buildFields() {
                const f = {
                    model: this.modelId,
                    prompt: this.prompt.trim(),
                    n: this.n,
                    output_format: this.outputFormat,
                    response_format: 'b64_json',
                };
                if (!(this.mode === 'edit' && this.followReference)) f.size = `${this.width}x${this.height}`;
                if (this.steps) f.steps = this.steps;
                if (this.guidance !== null && this.guidance !== '') f.guidance = Number(this.guidance);
                if (this.seed !== '' && this.seed !== null) f.seed = Number(this.seed);
                if (this.caps.negative_prompt && this.negativePrompt.trim()) f.negative_prompt = this.negativePrompt.trim();
                if (this.stepCache) f.accel = { step_cache: true };
                if (this.mode === 'edit' && this.caps.strength && this.strength < 1) f.strength = this.strength;
                return f;
            },

            async submit() {
                if (!this.canSubmit) return;
                this.error = '';
                this.running = true;
                this.progress = { done: 0, total: (this.steps || this.defaults.steps || 0) * this.n, phase: 'queue' };
                const started = performance.now();
                this.elapsed = 0;
                this._timer = setInterval(() => { this.elapsed = (performance.now() - started) / 1000; }, 100);
                this._poll = setInterval(() => this.pollProgress(), 700);
                this._abort = new AbortController();
                const fields = this.buildFields();
                const mode = this.mode;
                try {
                    let resp;
                    if (mode === 'edit') {
                        const fd = new FormData();
                        for (const [k, v] of Object.entries(fields)) {
                            fd.append(k, typeof v === 'object' ? JSON.stringify(v) : String(v));
                        }
                        this.references.forEach((r, i) => fd.append('image[]', r.blob, r.name || `ref${i}.png`));
                        if (this.mask && this.caps.mask) fd.append('mask', this.mask.blob, 'mask.png');
                        resp = await fetch('/v1/images/edits', {
                            method: 'POST', headers: this.headers(), body: fd, signal: this._abort.signal,
                        });
                    } else {
                        resp = await fetch('/v1/images/generations', {
                            method: 'POST',
                            headers: this.headers({ 'Content-Type': 'application/json' }),
                            body: JSON.stringify(fields),
                            signal: this._abort.signal,
                        });
                    }
                    const body = await resp.json().catch(() => ({}));
                    if (!resp.ok) {
                        const msg = body?.error?.message || body?.detail || resp.statusText;
                        throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
                    }
                    await this.storeResults(body, fields, mode);
                    if (!this.lockSeed) this.seed = '';
                } catch (e) {
                    if (e.name === 'AbortError') this.error = t('images.status.cancelled');
                    else this.error = e.message || String(e);
                } finally {
                    clearInterval(this._timer);
                    clearInterval(this._poll);
                    this.running = false;
                    this.progress = null;
                    this._abort = null;
                    this.$nextTick(() => window.lucide && window.lucide.createIcons());
                }
            },

            cancel() {
                if (this._abort) this._abort.abort();
            },

            async pollProgress() {
                try {
                    const resp = await fetch('/admin/api/activity', { credentials: 'same-origin' });
                    if (!resp.ok) return;
                    const data = await resp.json();
                    const models = data?.active_models?.models || [];
                    const m = models.find((x) => x.id === this.modelId);
                    if (!m || !this.progress) return;
                    if (m.is_loading) {
                        this.progress.phase = 'loading';
                        return;
                    }
                    const act = (m.activities || []).find((a) => a.kind === 'image');
                    if (act) {
                        this.progress.phase = 'running';
                        this.progress.done = act.completed_items || 0;
                        this.progress.total = act.total_items || this.progress.total;
                    }
                } catch (e) { /* progress is best effort */ }
            },

            get progressPct() {
                const p = this.progress;
                if (!p || !p.total) return 0;
                return Math.min(100, Math.round((p.done / p.total) * 100));
            },
            get progressText() {
                const p = this.progress;
                if (!p) return '';
                if (p.phase === 'loading') return t('images.status.loading');
                if (p.phase === 'queue') return t('images.status.starting');
                if (p.done >= p.total && p.total) return t('images.status.decoding');
                return t('images.status.step', { done: p.done, total: p.total });
            },

            async storeResults(body, fields, mode) {
                const mime = `image/${body.output_format || fields.output_format}`;
                const [w, h] = (body.size || `${this.width}x${this.height}`).split('x').map(Number);
                const created = Date.now();
                const perImage = body.usage ? body.usage.seconds / Math.max(1, (body.data || []).length) : null;
                const added = [];
                for (const [i, d] of (body.data || []).entries()) {
                    const item = {
                        id: `${created}-${i}-${Math.random().toString(36).slice(2, 8)}`,
                        created,
                        model: fields.model,
                        mode,
                        prompt: fields.prompt,
                        negative_prompt: fields.negative_prompt || '',
                        seed: d.seed,
                        width: w,
                        height: h,
                        steps: body.usage?.steps,
                        guidance: fields.guidance ?? null,
                        seconds: perImage,
                        format: body.output_format || fields.output_format,
                        references: mode === 'edit' ? this.references.length : 0,
                        blob: b64ToBlob(d.b64_json, mime),
                    };
                    await dbTx('readwrite', (s) => s.put(item));
                    added.push(item);
                }
                this.items = [...added, ...this.items];
                this.current = added[0] || this.current;
            },

            // ---- gallery --------------------------------------------------
            async loadGallery() {
                try {
                    const all = await dbTx('readonly', (s) => s.getAll());
                    this.items = (all || []).sort((a, b) => b.created - a.created);
                    this.current = this.items[0] || null;
                } catch (e) {
                    this.items = [];
                }
            },
            url(item) {
                if (!item) return '';
                if (!this.urls[item.id]) this.urls[item.id] = URL.createObjectURL(item.blob);
                return this.urls[item.id];
            },
            async deleteItem(item) {
                await dbTx('readwrite', (s) => s.delete(item.id));
                if (this.urls[item.id]) URL.revokeObjectURL(this.urls[item.id]);
                delete this.urls[item.id];
                this.items = this.items.filter((x) => x.id !== item.id);
                if (this.current && this.current.id === item.id) this.current = this.items[0] || null;
                if (this.lightbox && this.lightbox.id === item.id) this.lightbox = null;
            },
            async clearGallery() {
                if (!confirm(t('images.gallery.clear_confirm'))) return;
                await dbTx('readwrite', (s) => s.clear());
                Object.values(this.urls).forEach((u) => URL.revokeObjectURL(u));
                this.urls = {};
                this.items = [];
                this.current = null;
            },
            download(item) {
                const a = document.createElement('a');
                a.href = this.url(item);
                a.download = `omlx-${item.model}-${item.seed}.${item.format === 'jpeg' ? 'jpg' : item.format}`;
                a.click();
            },
            reuse(item) {
                if (this.models.some((m) => m.id === item.model)) this.modelId = item.model;
                this.prompt = item.prompt;
                this.negativePrompt = item.negative_prompt || '';
                this.seed = item.seed;
                this.lockSeed = true;
                if (item.steps) this.steps = item.steps;
                if (item.guidance !== null && item.guidance !== undefined) this.guidance = item.guidance;
                this.customSize = true;
                this.width = item.width;
                this.height = item.height;
                this.lightbox = null;
            },
            async useAsReference(item) {
                if (!this.canEdit) {
                    const editor = this.models.find((m) => m.capabilities?.edit);
                    if (!editor) {
                        this.error = t('images.error.no_editor');
                        return;
                    }
                    this.modelId = editor.id;
                }
                this.mode = 'edit';
                await this.addReferenceBlob(item.blob, `${item.seed}.${item.format}`);
                this.lightbox = null;
            },
            async copyPrompt(item) {
                try {
                    await navigator.clipboard.writeText(item.prompt);
                } catch (e) { /* ignore */ }
            },
            fmtTime(ts) {
                return new Date(ts).toLocaleString();
            },
        };
    };
})();
