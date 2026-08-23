<script setup lang="ts">
import { computed, onMounted, ref } from 'vue'

/**
 * 共享开关。
 *
 * 关掉时后端只监听一个 `0600` 的 Unix socket —— 登录节点上十几个别的用户
 * **连都连不上**（内核按文件权限拦）。打开才额外起一个 TCP 监听。
 * 所以这个开关不是「藏起来」，是真的有和没有。
 *
 * 共享出去的链接是**只读**的：访客看得到结果，但提交计算会被挡掉，
 * 不然任何拿到链接的人都能用本人的账号往 SLURM 投作业。
 */
interface ShareState {
  on: boolean
  addr: string | null
  host: string
  visitor: boolean
  idle_min: number
}

const s = ref<ShareState | null>(null)
const busy = ref(false)
const err = ref<string | null>(null)
const open = ref(false)
const copied = ref(false)

/** 从共享链接进来的人，整个开关都不该看到 */
const isVisitor = computed(() => s.value?.visitor === true)
const on = computed(() => s.value?.on === true)

/** 给别人的地址。后端 bind 的是 0.0.0.0，要换成访客能解析的主机名 */
const link = computed(() => {
  if (!s.value?.on || !s.value.addr) return ''
  const port = s.value.addr.split(':').pop()
  return `http://${s.value.host}:${port}`
})

async function load() {
  try {
    const r = await fetch('/api/share')
    if (r.ok) s.value = (await r.json()) as ShareState
  } catch {
    /* 拿不到就当没有，不弹错 */
  }
}

async function set(next: boolean) {
  busy.value = true
  err.value = null
  try {
    const r = await fetch('/api/share', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ on: next }),
    })
    if (!r.ok) throw new Error((await r.text()) || `HTTP ${r.status}`)
    const j = await r.json()
    s.value = { ...(s.value as ShareState), ...j }
    open.value = next
  } catch (e) {
    err.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function copy() {
  try {
    await navigator.clipboard.writeText(link.value)
    copied.value = true
    setTimeout(() => (copied.value = false), 1500)
  } catch {
    /* 剪贴板要权限，失败就算了 —— 地址本来就显示在那儿 */
  }
}

onMounted(load)
</script>

<template>
  <div v-if="s && !isVisitor" class="wrap">
    <button
      class="sw"
      :class="{ on }"
      :disabled="busy"
      :aria-pressed="on"
      @click="on ? set(false) : set(true)"
    >
      <span class="track"><span class="knob" /></span>
      <span class="t-small lbl">{{ on ? 'Shared' : 'Private' }}</span>
    </button>

    <button v-if="on" class="info" :title="'Sharing details'" @click="open = !open">ⓘ</button>

    <!-- 打开后的说明面板 -->
    <div v-if="open && on" class="pop">
      <p class="t-small">Give this address to someone else — anyone on the same network can open it:</p>
      <div class="row">
        <code class="mono addr">{{ link }}</code>
        <button class="btn btn-ghost tiny" @click="copy">{{ copied ? 'Copied' : 'Copy' }}</button>
      </div>
      <p class="t-small note">
        The shared link is <strong>read-only</strong> — visitors can see results but cannot
        submit computations. Switch it off and the listener disappears at once; nobody can connect.
      </p>
      <p v-if="s?.idle_min" class="t-small note">
        The backend stops automatically after <strong>{{ s.idle_min }} minutes</strong> idle,
        and sharing stops with it — it runs on a shared login node, so no process is left unattended.
      </p>
    </div>
    <p v-if="err" class="t-small err">{{ err }}</p>
  </div>
</template>

<style scoped>
.wrap {
  position: relative;
  display: flex;
  align-items: center;
  gap: 0.375rem;
}

.sw {
  display: inline-flex;
  align-items: center;
  gap: 0.5rem;
  background: none;
  border: none;
  padding: 0.25rem;
  color: var(--fg-3);
  font: inherit;
  cursor: pointer;
}
.sw:disabled {
  opacity: 0.5;
  cursor: default;
}
.track {
  width: 32px;
  height: 18px;
  border-radius: 999px;
  background: var(--fill-strong, #3a3a3a);
  border: 1px solid var(--line-strong);
  position: relative;
  transition: background-color 0.16s ease, border-color 0.16s ease;
  flex: none;
}
.knob {
  position: absolute;
  top: 2px;
  left: 2px;
  width: 12px;
  height: 12px;
  border-radius: 50%;
  background: var(--fg-3);
  transition: transform 0.16s ease, background-color 0.16s ease;
}
.sw.on .track {
  background: var(--green);
  border-color: var(--green);
}
.sw.on .knob {
  transform: translateX(14px);
  background: var(--on-green);
}
.sw.on .lbl {
  color: var(--green);
}
.lbl {
  white-space: nowrap;
}

.info {
  width: 20px;
  height: 20px;
  border: none;
  background: none;
  color: var(--fg-3);
  cursor: pointer;
  font-size: 0.8125rem;
  line-height: 1;
}
.info:hover {
  color: var(--fg-2);
}

.pop {
  position: absolute;
  top: calc(100% + 0.5rem);
  right: 0;
  z-index: 50;
  width: min(24rem, 80vw);
  padding: 0.875rem;
  background: var(--surface);
  border: 1px solid var(--line-strong);
  border-radius: var(--r-panel);
  box-shadow: 0 8px 32px rgba(0, 0, 0, 0.5);
}
.pop p {
  color: var(--fg-2);
  line-height: 1.6;
}
.row {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin: 0.5rem 0;
}
.addr {
  flex: 1;
  min-width: 0;
  padding: 0.375rem 0.5rem;
  background: var(--surface-2);
  border: 1px solid var(--line);
  border-radius: var(--r-ctl);
  font-size: 0.75rem;
  color: var(--fg);
  overflow-x: auto;
  white-space: nowrap;
}
.btn.tiny {
  height: 28px;
  padding: 0 0.625rem;
  font-size: 0.75rem;
  flex: none;
}
.note {
  color: var(--fg-3) !important;
}
.note strong {
  color: var(--fg-2);
}
.err {
  position: absolute;
  top: calc(100% + 0.25rem);
  right: 0;
  color: var(--red);
  white-space: nowrap;
}
</style>
