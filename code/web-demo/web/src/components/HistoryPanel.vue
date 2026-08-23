<script setup lang="ts">
import { STAGES, STATUS_LABEL, statusColor, stageLabel, fmtWhen } from '../lib/api'
import type { Submission } from '../types/Submission'

defineProps<{ subs: Submission[]; syncing: boolean }>()
const emit = defineEmits<{ detail: [Submission] }>()

/** 三段里跑到第几段了，画成进度条 */
function progress(s: Submission): number {
  const done = s.stages.filter((x) => x.state === 'COMPLETED').length
  return s.stages.length ? (done / s.stages.length) * 100 : 0
}

function stageIndex(s: Submission): number {
  return s.stages.findIndex((x) => x.kind === s.stage)
}

/** 为什么现在看不了详情。算完之前这里替代「详情」按钮。 */
function whyNoDetail(s: Submission): string {
  if (s.status === 'queued') return 'Queued — results appear once it finishes'
  if (s.status === 'running') {
    const cur = STAGES.find((x) => x.kind === s.stage)
    return cur ? `${cur.label} in progress` : 'Running — results appear once it finishes'
  }
  if (s.status === 'failed') return 'Failed — no output'
  if (s.status === 'cancelled') return 'Cancelled — no output'
  return ''
}
</script>

<template>
  <div class="panel">
    <div class="head">
      <h2 class="t-section">Submissions</h2>
      <!-- 记录本身来自本地文件，永远先显示出来；这里只说明状态可能还是上一次的 -->
      <span v-if="syncing" class="sync t-small"><i class="spin" />Refreshing status</span>
    </div>

    <div v-if="!subs.length" class="empty">
      <p class="t-body">No submissions yet</p>
      <p class="t-small">
        Submit a wave height on the left and every run shows up here with the stage it has
        reached.<br />
        Records are kept locally and survive a page refresh or a server restart.
      </p>
    </div>

    <ul v-else class="list">
      <li v-for="s in subs" :key="s.id" class="row" :class="s.status">
        <div class="head">
          <span class="dot" :style="{ background: statusColor(s.status) }" />
          <span class="h tnum">H = {{ (s.amp * 200).toFixed(1) }} cm</span>
          <span class="mono case">{{ s.case }}</span>
          <span class="when t-small tnum">{{ fmtWhen(s.created_at) }}</span>
        </div>

        <div class="body">
          <div class="stat">
            <span class="t-label" :style="{ color: statusColor(s.status) }">
              {{ STATUS_LABEL[s.status] ?? s.status }}
            </span>
            <span v-if="s.stage" class="t-small">· {{ stageLabel(s.stage) }}</span>
            <span v-if="s.reused_funwave" class="tag">FUNWAVE reused</span>
          </div>

          <!-- 三段的进度。跳过的那段画成空心，不占进度 -->
          <div class="track">
            <div class="fill" :style="{ width: progress(s) + '%', background: statusColor(s.status) }" />
          </div>
          <div class="legs t-small">
            <span
              v-for="(st, i) in STAGES"
              :key="st.kind"
              class="leg"
              :class="{
                gone: !s.stages.some((x) => x.kind === st.kind),
                done: s.stages.find((x) => x.kind === st.kind)?.state === 'COMPLETED',
                now: s.stage === st.kind,
                bad: s.status === 'failed' && s.stage === st.kind,
              }"
              :title="s.stages.find((x) => x.kind === st.kind)?.job_id ?? 'This stage was skipped'"
            >{{ i + 1 }}. {{ st.label }}</span>
          </div>
        </div>

        <div class="foot">
          <span class="t-small mono jobs">
            {{ s.stages.map((x) => x.job_id).join(' → ') }}
          </span>
          <!-- 没算完就**不渲染按钮**，只说一句为什么 ——
               一个点了没反应的禁用按钮比一行说明更让人困惑 -->
          <button v-if="s.status === 'done'" class="btn btn-ghost sm" @click="emit('detail', s)">
            Details
          </button>
          <span v-else class="t-small pend">{{ whyNoDetail(s) }}</span>
        </div>

        <p v-if="s.status === 'failed'" class="t-small err">
          Stage {{ stageIndex(s) + 1 }} failed. On the cluster, check the log for that job id
          under <span class="mono">code/logs/</span>.
        </p>
      </li>
    </ul>
  </div>
</template>

<style scoped>
.panel {
  padding: 0 1.75rem 1.75rem;
  min-width: 0;
}
.head {
  display: flex;
  align-items: baseline;
  gap: 0.75rem;
  margin-bottom: 1.5rem;
}
.sync {
  display: inline-flex;
  align-items: center;
  gap: 0.375rem;
  color: var(--fg-3);
}
.spin {
  width: 9px;
  height: 9px;
  border-radius: 50%;
  border: 1.5px solid var(--line-strong);
  border-top-color: var(--green);
  animation: spin 0.7s linear infinite;
}
@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}
@media (prefers-reduced-motion: reduce) {
  .spin {
    animation: none;
    border-top-color: var(--line-strong);
  }
}

.empty {
  border: 1px dashed var(--line);
  border-radius: var(--r-panel);
  padding: 2.5rem 1.5rem;
  text-align: center;
}
.empty p:first-child {
  margin-bottom: 0.5rem;
}
.empty .t-small {
  line-height: 1.7;
}

.list {
  list-style: none;
  display: grid;
  gap: 0.75rem;
}
.row {
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
  padding: 0.875rem 1rem;
  background: var(--surface-2);
}
.row.done {
  border-color: var(--line-strong);
}
.row.failed {
  border-color: color-mix(in srgb, var(--red) 45%, var(--line));
}

.head {
  display: flex;
  align-items: baseline;
  gap: 0.5rem;
  flex-wrap: wrap;
}
.dot {
  width: 7px;
  height: 7px;
  border-radius: 50%;
  flex: none;
  align-self: center;
}
.h {
  font-size: 1rem;
  font-weight: 500;
}
.case {
  font-size: 0.75rem;
  color: var(--fg-3);
}
.when {
  margin-left: auto;
}

.body {
  margin-top: 0.625rem;
}
.stat {
  display: flex;
  align-items: baseline;
  gap: 0.375rem;
  flex-wrap: wrap;
  margin-bottom: 0.5rem;
}
.tag {
  margin-left: 0.25rem;
  padding: 0.0625rem 0.375rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-ctl);
  font-size: 0.6875rem;
  color: var(--fg-3);
}

.track {
  height: 3px;
  border-radius: 2px;
  background: var(--line);
  overflow: hidden;
}
.fill {
  height: 100%;
  transition: width 0.3s ease;
}
.legs {
  display: flex;
  gap: 0.875rem;
  flex-wrap: wrap;
  margin-top: 0.4375rem;
}
.leg {
  color: var(--fg-3);
}
.leg.done {
  color: var(--fg-2);
}
.leg.now {
  color: var(--green);
}
.leg.bad {
  color: var(--red);
}
.leg.gone {
  text-decoration: line-through;
  opacity: 0.5;
}

.foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 0.75rem;
  margin-top: 0.75rem;
}
.jobs {
  color: var(--fg-3);
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.pend {
  flex: none;
  color: var(--fg-3);
  white-space: nowrap;
}
.btn.sm {
  height: 30px;
  padding: 0 0.875rem;
  font-size: 0.8125rem;
  flex: none;
}
.err {
  color: var(--red);
  margin-top: 0.5rem;
  line-height: 1.6;
}
</style>
