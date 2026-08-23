import { ref } from 'vue'
import type { Submission } from '../types/Submission'
import type { SubmissionList } from '../types/SubmissionList'

/** 后端 /api/config 吐的锁定配置 */
/** `results/web/model/` 下扫到的一个 checkpoint */
export interface Checkpoint {
  run_name: string
  run_ts: string
  file: string
  bytes: number
  current: boolean
}

export interface Config {
  run_name: string
  run_ts: string
  case: string
  amp: number
  slope_label: string
  chunk: number
  fields: string[]
  /** 后端每次启动扫盘得到的，不是前端写死的 */
  checkpoints: Checkpoint[]
}

/**
 * 算例目录、锁定配置、提交与提交记录。
 *
 * 全是显式拉取，**没有 SSE、没有常驻轮询**：目录只在跑完一次渲染时才变，
 * 提交记录只在有作业在跑的时候才需要刷。
 */
export function useApi() {
  const config = ref<Config | null>(null)
  const subs = ref<Submission[]>([])
  /** 后端正在跟集群对齐。记录**已经显示出来了**，只是状态可能还是上一次的 */
  const syncing = ref(false)
  /** 上一次成功对齐的时刻（unix 秒） */
  const syncedAt = ref<number | null>(null)
  /** 连不上后端。后端连不上集群是另一回事，那时它会沿用上次的状态继续服务 */
  const error = ref<string | null>(null)
  /** 提交失败。与 error 分开 —— 一个是「看不到」，一个是「投不出去」 */
  const submitError = ref<string | null>(null)
  const submitting = ref(false)

  async function loadConfig() {
    try {
      const r = await fetch('/api/config')
      if (r.ok) config.value = (await r.json()) as Config
    } catch {
      /* 配置拿不到时前端有一套内置默认值兜着，不至于白屏 */
    }
  }

  /**
   * 这个接口顺手把还活着的作业状态刷一遍，所以轮询只轮它一个。
   *
   * `background = true` 时带上 `x-wave-poll` —— 告诉后端这是机器在轮不是人在动，
   * 别拿它去续闲置自停的命。忘了带的话，一个开着没人管的标签页会把后端一直吊着。
   */
  async function loadSubs(background = false) {
    try {
      const r = await fetch('/api/submissions', {
        headers: background ? { 'x-wave-poll': '1' } : {},
      })
      if (!r.ok) throw new Error(`HTTP ${r.status}`)
      const j = (await r.json()) as SubmissionList
      subs.value = j.items
      syncing.value = j.syncing
      syncedAt.value = j.synced_at
      error.value = null
    } catch (e) {
      // 沿用上一次的列表，不清空 —— 网络抖一下不该让页面变空
      error.value = e instanceof Error ? e.message : String(e)
    }
  }

  /** 不带参数 —— 后端按锁定的默认配置跑 */
  async function submit(): Promise<Submission | null> {
    submitting.value = true
    submitError.value = null
    try {
      const r = await fetch('/api/submit', { method: 'POST' })
      // 集群投不出去是 502，正文是纯文本，直接显示比包一层 JSON 有用
      if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`)
      const s = (await r.json()) as Submission
      subs.value = [s, ...subs.value]
      return s
    } catch (e) {
      submitError.value = e instanceof Error ? e.message : String(e)
      return null
    } finally {
      submitting.value = false
    }
  }

  return {
    config,
    subs,
    syncing,
    syncedAt,
    error,
    submitError,
    submitting,
    loadConfig,
    loadSubs,
    submit,
  }
}

export function videoUrl(stage: string, name: string, chunk: number, field: string): string {
  return `/api/video/${stage}/${name}/${chunk}/${field}`
}

/** 通道 → 人话。OpenFOAM 的场名直接摆出来没人看得懂。 */
export const FIELD_LABEL: Record<string, string> = {
  alpha: 'Free surface',
  Ux: 'Horizontal velocity',
  Uz: 'Vertical velocity',
  p_rgh: 'Dynamic pressure',
}

export const FIELD_SUB: Record<string, string> = {
  alpha: 'alpha.water',
  Ux: 'αUx',
  Uz: 'αUz',
  p_rgh: 'p_rgh',
}

/** 三段流水线 */
export const STAGES = [
  { kind: 'funwave', label: 'FUNWAVE solve', sub: '2D Boussinesq' },
  { kind: 'prior', label: 'Lift to 3D prior', sub: 'Nwogu profile' },
  { kind: 'vis', label: 'HPM rollout + render', sub: '1000 frames · four channels' },
] as const

export function stageLabel(kind: string): string {
  return STAGES.find((s) => s.kind === kind)?.label ?? kind
}

export const STATUS_LABEL: Record<string, string> = {
  queued: 'Queued',
  running: 'Running',
  done: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
}

export function statusColor(s: string): string {
  if (s === 'running') return 'var(--green)'
  if (s === 'queued') return 'var(--amber)'
  if (s === 'done') return 'var(--fg-2)'
  return 'var(--red)'
}

/** 还在跑 —— 用来决定要不要继续轮询 */
export function isLive(s: string): boolean {
  return s === 'queued' || s === 'running'
}

export function fmtWhen(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: 'numeric',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    hour12: false,
  })
}
