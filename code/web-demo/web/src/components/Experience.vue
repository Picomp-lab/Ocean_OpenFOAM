<script setup lang="ts">
import { computed, onMounted, onUnmounted, ref, watch } from 'vue'
import InputPanel from './InputPanel.vue'
import HistoryPanel from './HistoryPanel.vue'
import DetailModal from './DetailModal.vue'
import { useApi, isLive } from '../lib/api'
import { useActivity } from '../lib/useActivity'
import type { Submission } from '../types/Submission'

const {
  config,
  subs,
  syncing,
  error,
  submitError,
  submitting,
  loadConfig,
  loadSubs,
  submit,
} = useApi()

/** 弹窗。case_/amp 两种来源共用一套字段，见 DetailModal 的注释 */
const modalOpen = ref(false)
const modalCase = ref('')
const modalAmp = ref(0)
const modalSub = ref<Submission | null>(null)

let timer: number | undefined

// 真人点击/按键时告诉后端「还有人在」，后台轮询不算
useActivity()

const anyLive = computed(() => subs.value.some((s) => isLive(s.status)))

async function onSubmit() {
  const s = await submit()
  if (s) poll()
}

function openSub(s: Submission) {
  modalCase.value = s.case
  modalAmp.value = s.amp
  modalSub.value = s
  modalOpen.value = true
}

/**
 * 只在**自己有作业在跑**的时候才轮询，跑完立刻停。
 * 演示页放在那儿没人动的时候，它对集群是完全静默的。
 */
async function poll() {
  await loadSubs(true) // 后台轮询：不刷新后端的闲置计时
  schedule()
}

function schedule() {
  clearTimeout(timer)
  // 15 秒足够：链路里最短的一段也要 6 分钟，再密没有意义。
  // 后端正在对齐时早一点回来取结果 —— 那一次是后台跑的，很快就有。
  if (anyLive.value) timer = window.setTimeout(poll, syncing.value ? 2000 : 15000)
}

watch(anyLive, schedule)

onMounted(async () => {
  await Promise.all([loadConfig(), loadSubs()])
  schedule()
})

onUnmounted(() => clearTimeout(timer))
</script>

<template>
  <div v-if="error" class="warn">
    <p class="t-small">Cannot reach the backend; showing the last records retrieved. <span class="mono">{{ error }}</span></p>
  </div>

  <div class="split">
    <InputPanel
      :config="config"
      :submitting="submitting || syncing"
      :submit-error="submitError"
      @submit="onSubmit"
    />
    <HistoryPanel :subs="subs" :syncing="syncing" @detail="openSub" />
  </div>

  <DetailModal
    :open="modalOpen"
    :case_="modalCase"
    :amp="modalAmp"
    :chunk="config?.chunk ?? 10"
    :fields="config?.fields ?? ['alpha', 'Ux', 'Uz', 'p_rgh']"
    :sub="modalSub"
    @close="modalOpen = false"
  />

</template>

<style scoped>
.warn {
  margin: 0 1.75rem 1rem;
  padding: 0.625rem 1rem;
  border-left: 2px solid var(--amber);
  background: var(--surface-2);
  border-radius: var(--r-ctl);
}
.warn p {
  color: var(--fg-2);
}
.warn .mono {
  color: var(--fg-3);
  word-break: break-all;
}

/* 输入 | 提交记录，中间一条竖线。窄屏落成上下两段。 */
.split {
  display: grid;
  grid-template-columns: 1fr;
  margin-top: 0.75rem;
}
.split > :first-child {
  border-bottom: 1px solid var(--line);
  padding-bottom: 1.75rem;
}
@media (min-width: 60rem) {
  .split {
    grid-template-columns: minmax(0, 23rem) minmax(0, 1fr);
  }
  .split > :first-child {
    border-bottom: none;
    border-right: 1px solid var(--line);
    padding-bottom: 0;
  }
}
@media (min-width: 78rem) {
  .split {
    grid-template-columns: minmax(0, 25rem) minmax(0, 1fr);
  }
}

/* 只给读屏用 */
.sr {
  position: absolute;
  width: 1px;
  height: 1px;
  overflow: hidden;
  clip: rect(0 0 0 0);
}
</style>
