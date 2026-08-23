<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import { videoUrl, FIELD_LABEL, FIELD_SUB, stageLabel, fmtWhen } from '../lib/api'
import type { Submission } from '../types/Submission'

/**
 * 结果弹窗。两种打开方式：
 *   - 从提交记录点「详情」→ 带 `sub`，能显示这次跑了哪三段、各段作业号
 *   - 从滑块上的已有结果点开 → 只有 case/amp，没有流水线信息
 */
const props = defineProps<{
  open: boolean
  case_: string
  amp: number
  chunk: number
  fields: string[]
  sub: Submission | null
}>()

const emit = defineEmits<{ close: [] }>()

const field = ref('alpha')
const src = computed(() =>
  props.open ? videoUrl('lt', props.case_, props.chunk, field.value) : null,
)

/**
 * 解码看门狗 —— 与 OutputPanel 里同一个理由：3840×1080 的 H.264 偶尔会
 * **既不出帧也不报错**（readyState 停在 0、error 是 null），页面上就是个
 * 转不完的圈。超时就换成说明 + 直接打开文件的链接。
 */
const STALL_MS = 12000
const stalled = ref(false)
const failed = ref(false)
let timer: number | undefined

function arm() {
  clearTimeout(timer)
  stalled.value = false
  failed.value = false
  if (src.value) timer = window.setTimeout(() => (stalled.value = true), STALL_MS)
}
function onReady() {
  clearTimeout(timer)
  stalled.value = false
}
watch(src, arm, { immediate: true })
// 每次打开都从自由面看起 —— 它是最直观的那个通道
watch(
  () => props.open,
  (o) => {
    if (o) field.value = 'alpha'
  },
)

function onKey(e: KeyboardEvent) {
  if (e.key === 'Escape' && props.open) emit('close')
}
onMounted(() => window.addEventListener('keydown', onKey))
onUnmounted(() => {
  window.removeEventListener('keydown', onKey)
  clearTimeout(timer)
})
</script>

<template>
  <Teleport to="body">
    <div v-if="open" class="scrim" @click.self="emit('close')">
      <div class="modal" role="dialog" aria-modal="true">
        <header class="head">
          <div class="titles">
            <h2 class="t-section">H = {{ (amp * 200).toFixed(1) }} cm</h2>
            <p class="t-small">
              <span class="mono">{{ case_ }}</span>
              · Slope 1:35 · T = 2 s · d = 40 cm
              <template v-if="sub"> · Submitted {{ fmtWhen(sub.created_at) }}</template>
            </p>
          </div>
          <button class="x" aria-label="Close" @click="emit('close')">✕</button>
        </header>

        <!-- 这次跑了哪几段。从滑块点开的没有这块 -->
        <div v-if="sub" class="stages">
          <div v-for="st in sub.stages" :key="st.kind" class="st">
            <span class="t-small nm">{{ stageLabel(st.kind) }}</span>
            <span class="t-small mono jid">{{ st.job_id }}</span>
            <span class="t-small el tnum">{{ st.elapsed || '—' }}</span>
          </div>
          <p v-if="sub.reused_funwave" class="t-small note">
            The FUNWAVE stage reused output already on the cluster; it was not re-run.
          </p>
        </div>

        <div class="seg" role="tablist">
          <button
            v-for="f in fields"
            :key="f"
            class="seg-btn"
            :class="{ on: field === f }"
            role="tab"
            :aria-selected="field === f"
            @click="field = f"
          >
            <span class="nm">{{ FIELD_LABEL[f] ?? f }}</span>
            <span class="mono sub">{{ FIELD_SUB[f] ?? f }}</span>
          </button>
        </div>

        <div class="stage-box">
          <video
            v-if="src"
            :key="src"
            :src="src"
            class="vid"
            controls
            autoplay
            loop
            muted
            playsinline
            @loadeddata="onReady"
            @canplay="onReady"
            @error="failed = true"
          />
          <div v-if="stalled || failed" class="stall">
            <p class="t-body">Your browser could not decode this video</p>
            <p class="t-small">
              The file itself is fine — the server already fetched it. The local media decoder
              simply produced no frames. Restarting the browser usually fixes it, or you can open
              the raw file directly.
            </p>
            <a v-if="src" class="btn btn-ghost" :href="src" target="_blank" rel="noopener">
              Open the mp4 in a new tab
            </a>
          </div>
        </div>

        <footer class="foot">
          <p class="t-small dim">
            y = 0.30 m slice · 107,466 cells · chunk {{ chunk }} · 1000 frames · Δt = 0.05 s ·
            HPM long-horizon rollout, no CFD ground truth for comparison
          </p>
        </footer>
      </div>
    </div>
  </Teleport>
</template>

<style scoped>
.scrim {
  position: fixed;
  inset: 0;
  z-index: 100;
  background: rgba(0, 0, 0, 0.72);
  display: flex;
  align-items: flex-start;
  justify-content: center;
  padding: 3rem 1.5rem;
  overflow-y: auto;
}
.modal {
  width: min(64rem, 100%);
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-card);
  padding: 1.5rem;
}

.head {
  display: flex;
  align-items: flex-start;
  justify-content: space-between;
  gap: 1rem;
  margin-bottom: 1.25rem;
}
.titles p {
  color: var(--fg-3);
  margin-top: 0.25rem;
}
.x {
  flex: none;
  width: 32px;
  height: 32px;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-ctl);
  background: transparent;
  color: var(--fg-2);
  font-size: 0.875rem;
  cursor: pointer;
}
.x:hover {
  background: var(--surface-2);
  color: var(--fg);
}

/* ---------- 流水线 ---------- */
.stages {
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
  padding: 0.75rem 0.875rem;
  margin-bottom: 1.25rem;
  background: var(--surface-2);
}
.st {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  padding: 0.1875rem 0;
}
.st .nm {
  flex: 1;
  color: var(--fg-2);
}
.st .jid {
  color: var(--fg-3);
}
.st .el {
  color: var(--fg-3);
  width: 5rem;
  text-align: right;
}
.note {
  color: var(--fg-3);
  line-height: 1.6;
}

/* ---------- 通道分段 ---------- */
.seg {
  display: flex;
  gap: 2px;
  padding: 3px;
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
  margin-bottom: 1rem;
  overflow-x: auto;
}
.seg-btn {
  flex: 1 1 0;
  min-width: 7rem;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.0625rem;
  padding: 0.5rem 0.625rem;
  background: transparent;
  border: 1px solid transparent;
  border-radius: calc(var(--r-panel) - 3px);
  color: var(--fg-2);
  font: inherit;
  cursor: pointer;
  transition:
    background-color 0.12s ease,
    border-color 0.12s ease,
    color 0.12s ease;
}
.seg-btn:hover:not(.on) {
  background: var(--surface-2);
}
.seg-btn.on {
  border-color: var(--fg);
  color: var(--fg);
}
.seg-btn .nm {
  font-size: 0.875rem;
}
.seg-btn .sub {
  font-size: 0.6875rem;
  color: var(--fg-3);
}

/* ---------- 画面 ---------- */
.stage-box {
  position: relative;
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
  overflow: hidden;
  background: #000;
  min-height: 10rem;
}
.vid {
  width: 100%;
  display: block;
}
.stall {
  position: absolute;
  inset: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 0.625rem;
  text-align: center;
  padding: 2rem 1.5rem;
  background: color-mix(in srgb, var(--surface-2) 94%, transparent);
}
.stall p {
  max-width: 30rem;
  line-height: 1.6;
}
.stall .t-small {
  color: var(--fg-3);
}
.stall .btn {
  margin-top: 0.25rem;
  text-decoration: none;
}

.foot {
  margin-top: 1rem;
  padding-top: 0.875rem;
  border-top: 1px solid var(--line);
  display: grid;
  gap: 0.375rem;
}
.dim {
  color: var(--fg-3);
}
</style>
