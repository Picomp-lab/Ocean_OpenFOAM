<script setup lang="ts">
import { computed, ref } from 'vue'
import { STAGES } from '../lib/api'
import type { Config } from '../lib/api'

/**
 * 配置面板。
 *
 * 控件是**真的画出来的**，不是灰掉的占位 —— 让人看见这套东西以后能调什么。
 * 但这一版只允许默认配置：动任何一个控件都会弹警示并把值弹回去。
 * 解锁时删掉 `lockedTry` 的调用、接上真的 v-model 就行，布局不用动。
 */
const props = defineProps<{
  config: Config | null
  submitting: boolean
  submitError: string | null
}>()

const emit = defineEmits<{ submit: [] }>()

const amp = computed(() => props.config?.amp ?? 0.0635)
const heightCm = computed(() => amp.value * 200)

/**
 * 滑块的显示区间。**只是画给人看的** —— 这一版后端不收波高参数，
 * 所以范围写在前端。解锁时应该改成由 `/api/config` 给。
 */
const AMP_MIN = 0.03
const AMP_MAX = 0.075
const pct = computed(() => ((amp.value - AMP_MIN) / (AMP_MAX - AMP_MIN)) * 100)

/** 下拉里列的都是集群上真实存在的选项，不是编出来充数的 */
const SLOPES = [
  { v: '325', label: '1:32.5' },
  { v: '350', label: '1:35' },
  { v: '375', label: '1:37.5' },
]
const CHUNKS = [
  { v: '10', label: 'chunk 10 · t 50–100 s · 1000 frames' },
  { v: '9', label: 'chunk 9 · t 45–50 s · 100 frames' },
]
const ALL_FIELDS = [
  { v: 'alpha', label: 'alpha' },
  { v: 'Ux', label: 'Ux' },
  { v: 'Uz', label: 'Uz' },
  { v: 'p_rgh', label: 'p_rgh' },
  { v: 'Uy', label: 'Uy' },
  { v: 'nut', label: 'nut' },
]
const onFields = computed(() => props.config?.fields ?? ['alpha', 'Ux', 'Uz', 'p_rgh'])

/**
 * checkpoint 下拉列的是**后端每次启动扫 `results/web/model/` 得到的**，
 * 不是前端写死的 —— 写死的话会列出盘上根本没有的权重。
 */
const ckpts = computed(() =>
  (props.config?.checkpoints ?? []).map((c) => ({
    key: `${c.run_name}/${c.run_ts}/${c.file}`,
    label: `${c.run_name} · ${c.file}${c.current ? ' (current)' : ''}`,
  })),
)
const currentKey = computed(() => {
  const cur = (props.config?.checkpoints ?? []).find((c) => c.current)
  return cur ? `${cur.run_name}/${cur.run_ts}/${cur.file}` : ''
})

/** 弹出警示时说的是哪一项 */
const locked = ref<{ what: string; why: string } | null>(null)

const REASONS: Record<string, string> = {
  'Wave height':
    'The backend can already build a FUNWAVE case at any wave height and run the full ' +
    'pipeline, but that costs about 4 h 24 min more. This build stays fixed at the ' +
    'training condition.',
  'Bed slope':
    "The surrogate model's LBO spectral basis and coordinates are tied to the 1:35 CFD " +
    'mesh. Changing the slope only alters the FUNWAVE bed, so the two mismatch by ' +
    '1.3–3.4 cm — the backend accepts the parameter, but the results become markedly ' +
    'less trustworthy.',
  Checkpoint:
    'The pure HPM line (window=6) has no prior base and does not share the prior→vis ' +
    'interface of this pipeline; switching to it would mean wiring up a separate path.',
  'Time window':
    'chunk 10 is the only window that needs no t-offset calibration. Other chunks require ' +
    'a scan_toffset run first, and that path is not wired up here.',
  'Output channels':
    'Uy is close to noise in this quasi-2D case (model nRMSE 1.000 ≈ baseline), and nut ' +
    'does not structurally exist in an inviscid Boussinesq prior. Both were switched off ' +
    'during training — they are not locked here.',
}

/**
 * 任何一次改动都在这里被挡回去：把 DOM 的值弹回默认，再弹警示。
 *
 * 必须手动改 `el.value` —— 控件绑的是常量，响应式状态没变，Vue 不会重渲染，
 * 光弹窗的话控件会停在用户拖到的位置上。
 */
function lockedTry(e: Event, what: string, back: string | boolean) {
  const el = e.target as HTMLInputElement | HTMLSelectElement
  if (typeof back === 'boolean') (el as HTMLInputElement).checked = back
  else el.value = back
  locked.value = { what, why: REASONS[what] ?? '' }
}
</script>

<template>
  <div class="panel">
    <div class="head">
      <h2 class="t-section">Configuration</h2>
      <span class="pill">Default configuration · locked</span>
    </div>
    <p class="t-small lede">
      All of these will be adjustable later. <strong>This build allows only the default
      configuration</strong> — changing any of them is refused, with the reason given.
    </p>

    <!-- 波高 -->
    <div class="field">
      <span class="lab">Incident wave height H</span>
      <div class="readout">
        <span class="val tnum">{{ heightCm.toFixed(1) }}</span>
        <span class="unit">cm</span>
        <span class="aux t-small tnum mono">AMP_WK {{ amp.toFixed(4) }} m</span>
      </div>
      <input
        class="slider"
        type="range"
        :min="AMP_MIN"
        :max="AMP_MAX"
        step="0.0001"
        :value="amp"
        :style="{ '--pct': pct + '%' }"
        @input="lockedTry($event, 'Wave height', String(amp))"
      />
      <div class="scale t-small tnum">
        <span>{{ (AMP_MIN * 200).toFixed(0) }}</span>
        <span class="mid">Training condition {{ heightCm.toFixed(1) }} cm</span>
        <span>{{ (AMP_MAX * 200).toFixed(0) }}</span>
      </div>
    </div>

    <!-- 底坡 -->
    <div class="field">
      <span class="lab">Bed slope</span>
      <div class="radios">
        <button
          v-for="s in SLOPES"
          :key="s.v"
          class="radio"
          :class="{ on: s.label === (config?.slope_label ?? '1:35') }"
          @click="s.label !== (config?.slope_label ?? '1:35') && (locked = { what: 'Bed slope', why: REASONS['Bed slope'] })"
        >
          <i class="dot" /><span>{{ s.label }}</span>
        </button>
      </div>
    </div>

    <!-- checkpoint -->
    <div class="field">
      <span class="lab">Model checkpoint</span>
      <select
        class="select"
        :value="currentKey"
        @change="lockedTry($event, 'Checkpoint', currentKey)"
      >
        <option v-for="c in ckpts" :key="c.key" :value="c.key">{{ c.label }}</option>
        <option v-if="!ckpts.length" value="">(no .pt found under model/)</option>
      </select>
      <p class="t-small hint mono">{{ config?.run_ts ?? '—' }}</p>
    </div>

    <!-- 时间区间 -->
    <div class="field">
      <span class="lab">Time window</span>
      <select
        class="select"
        :value="String(config?.chunk ?? 10)"
        @change="lockedTry($event, 'Time window', String(config?.chunk ?? 10))"
      >
        <option v-for="c in CHUNKS" :key="c.v" :value="c.v">{{ c.label }}</option>
      </select>
    </div>

    <!-- 通道 -->
    <div class="field">
      <span class="lab">Output channels</span>
      <div class="chips">
        <label
          v-for="f in ALL_FIELDS"
          :key="f.v"
          class="chip"
          :class="{ on: onFields.includes(f.v) }"
        >
          <input
            type="checkbox"
            :checked="onFields.includes(f.v)"
            @change="lockedTry($event, 'Output channels', onFields.includes(f.v))"
          />
          <span class="mono">{{ f.label }}</span>
        </label>
      </div>
    </div>

    <!-- 提交后会跑 -->
    <div class="plan">
      <div class="lab">What runs after you submit</div>
      <ol class="steps">
        <li v-for="s in STAGES" :key="s.kind">
          <span class="n" />
          <div>
            <div class="t-label">
              {{ s.label }}
              <span v-if="s.kind === 'funwave'" class="skiptag">reuses existing</span>
            </div>
            <div class="t-small">{{ s.sub }}</div>
          </div>
        </li>
      </ol>
    </div>

    <button class="btn btn-primary go" :disabled="submitting" @click="emit('submit')">
      {{ submitting ? 'Please wait…' : 'Run with default configuration' }}
    </button>
    <p v-if="submitError" class="t-small err">{{ submitError }}</p>

    <!-- ── 锁定警示 ── -->
    <Teleport to="body">
      <div v-if="locked" class="scrim" @click.self="locked = null">
        <div class="warnbox" role="alertdialog" aria-modal="true">
          <div class="wtop">
            <svg class="wic" viewBox="0 0 20 20" aria-hidden="true">
              <path
                d="M10 2.5 18.5 17.5H1.5z"
                fill="none"
                stroke="currentColor"
                stroke-width="1.5"
                stroke-linejoin="round"
              />
              <path d="M10 8v4.2M10 14.6v.6" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" />
            </svg>
            <div>
              <h3 class="t-section">{{ locked.what }} is locked</h3>
              <p class="t-small sub">This build is fixed to the default configuration used in training.</p>
            </div>
          </div>
          <p class="t-small why">{{ locked.why }}</p>
          <button class="btn btn-primary" @click="locked = null">Got it</button>
        </div>
      </div>
    </Teleport>
  </div>
</template>

<style scoped>
.panel {
  padding: 0 1.75rem 1.75rem;
}
.head {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  margin-bottom: 0.75rem;
  flex-wrap: wrap;
}
.pill {
  padding: 0.0625rem 0.4375rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-ctl);
  font-size: 0.6875rem;
  color: var(--fg-3);
  white-space: nowrap;
}
.lede {
  line-height: 1.7;
  margin-bottom: 1.5rem;
}
.lede strong {
  color: var(--fg);
}
.lab {
  display: block;
  font-size: 0.875rem;
  margin-bottom: 0.5rem;
}
.hint {
  margin-top: 0.5rem;
  line-height: 1.6;
}
.hint strong {
  color: var(--fg-2);
}

/* ---------- 波高 ---------- */
.readout {
  display: flex;
  align-items: baseline;
  gap: 0.375rem;
  margin-bottom: 0.625rem;
}
.val {
  font-size: 2rem;
  font-weight: 500;
  line-height: 1;
}
.unit {
  font-size: 1rem;
  color: var(--fg-2);
}
.aux {
  margin-left: auto;
  color: var(--fg-3);
}

/* 轨道中线固定在距顶 0.625rem 处（input 高 1.25rem，WebKit 把 4px 轨道居中）。
   别给轨道加 margin-top —— 那会让把手和轨道各歪各的。 */
.slider {
  display: block;
  width: 100%;
  height: 1.25rem;
  appearance: none;
  background: transparent;
  cursor: pointer;
  margin: 0;
}
.slider::-webkit-slider-runnable-track {
  height: 4px;
  border-radius: 2px;
  background: linear-gradient(
    to right,
    var(--green) 0 var(--pct),
    var(--line-strong) var(--pct) 100%
  );
}
.slider::-moz-range-track {
  height: 4px;
  border-radius: 2px;
  background: linear-gradient(
    to right,
    var(--green) 0 var(--pct),
    var(--line-strong) var(--pct) 100%
  );
}
.slider::-webkit-slider-thumb {
  appearance: none;
  width: 14px;
  height: 14px;
  margin-top: -5px;
  border-radius: 50%;
  background: var(--green);
  border: 2px solid var(--bg);
}
.slider::-moz-range-thumb {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  background: var(--green);
  border: 2px solid var(--bg);
}
.scale {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 0.5rem;
  margin-top: 0.125rem;
}
.scale .mid {
  color: var(--fg-3);
}

/* ---------- 单选行 ---------- */
.radios {
  display: flex;
  flex-wrap: wrap;
  gap: 1.125rem;
}
.radio {
  display: inline-flex;
  align-items: center;
  gap: 0.4375rem;
  background: none;
  border: none;
  padding: 0.25rem 0;
  color: var(--fg-2);
  font: inherit;
  font-size: 0.875rem;
  cursor: pointer;
}
.radio .dot {
  width: 14px;
  height: 14px;
  border-radius: 50%;
  border: 1px solid var(--line-strong);
  flex: none;
}
.radio.on {
  color: var(--fg);
}
.radio.on .dot {
  border-color: var(--green);
  background: radial-gradient(circle, var(--green) 0 3.5px, transparent 3.5px);
}

/* ---------- 通道 ---------- */
.chips {
  display: flex;
  flex-wrap: wrap;
  gap: 0.375rem;
}
.chip {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  padding: 0.3125rem 0.6875rem;
  border: 1px solid var(--line);
  border-radius: var(--r-ctl);
  background: var(--surface-2);
  color: var(--fg-3);
  font-size: 0.8125rem;
  cursor: pointer;
}
.chip.on {
  border-color: var(--green);
  color: var(--fg);
}
.chip input {
  accent-color: var(--green);
  margin: 0;
  cursor: pointer;
}

/* ---------- 流水线预告 ---------- */
.plan {
  margin: 1.75rem 0 1.25rem;
}
.steps {
  list-style: none;
  display: grid;
  gap: 0.5rem;
}
.steps li {
  display: flex;
  gap: 0.625rem;
  align-items: flex-start;
}
.steps .n {
  flex: none;
  width: 7px;
  height: 7px;
  margin-top: 0.4375rem;
  border-radius: 50%;
  background: var(--green);
}
.skiptag {
  margin-left: 0.375rem;
  padding: 0.0625rem 0.375rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-ctl);
  font-size: 0.6875rem;
  color: var(--fg-3);
}

.go {
  width: 100%;
}
.err {
  color: var(--red);
  word-break: break-word;
  margin-top: 0.5rem;
}
/* ---------- 锁定警示 ---------- */
.scrim {
  position: fixed;
  inset: 0;
  z-index: 200;
  background: rgba(0, 0, 0, 0.72);
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.5rem;
}
.warnbox {
  width: min(30rem, 100%);
  background: var(--surface);
  border: 1px solid var(--line-strong);
  border-radius: var(--r-panel);
  padding: 1.375rem;
}
.wtop {
  display: flex;
  gap: 0.875rem;
  align-items: flex-start;
  margin-bottom: 0.875rem;
}
.wic {
  width: 20px;
  height: 20px;
  flex: none;
  margin-top: 0.1875rem;
  color: var(--amber);
}
.sub {
  color: var(--fg-3);
  margin-top: 0.1875rem;
}
.why {
  color: var(--fg-2);
  line-height: 1.7;
  padding: 0.75rem 0.875rem;
  border-left: 2px solid var(--line-strong);
  background: var(--surface-2);
  border-radius: 0 var(--r-ctl) var(--r-ctl) 0;
  margin-bottom: 1.125rem;
}
.warnbox .btn {
  width: 100%;
}
</style>
