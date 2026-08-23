<script setup lang="ts">
import { ref } from 'vue'
import Experience from './components/Experience.vue'
import ModelCard from './components/ModelCard.vue'
import ShareToggle from './components/ShareToggle.vue'

const TABS = [
  { id: 'demo', label: 'Interactive demo' },
  { id: 'card', label: 'How it works' },
] as const

const tab = ref<'demo' | 'card'>('demo')

const TAGS = [
  'cfd',
  'openfoam',
  'boussinesq',
  'funwave',
  'surrogate-model',
  'wave-breaking',
  'spectral-operator',
]
</script>

<template>
  <header class="topbar">
    <div class="inner">
      <div class="brand">
        <!-- 一道爬上斜坡的波：整个算例的形状 -->
        <svg class="mark" viewBox="0 0 32 20" aria-hidden="true">
          <path d="M1 15 L31 5" stroke="var(--fg-3)" stroke-width="1.5" stroke-linecap="round" />
          <path
            d="M1 11c2.6 0 2.6-4 5.2-4s2.6 4 5.2 4 2.6-5 5.2-5 2.6 5 5.2 5 2.6-6.5 5.2-6.5"
            fill="none"
            stroke="var(--green)"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
          />
        </svg>
        <span class="name">Wave Field Surrogate</span>
      </div>
      <div class="right">
        <span class="t-small host">Compute backend <span class="mono">OSU HPC</span></span>
        <ShareToggle />
      </div>
    </div>
  </header>

  <div class="hero">
    <div class="inner">
      <div class="htext">
        <div class="eyebrow t-small mono">oregon state university · college of engineering cluster</div>
        <h1 class="t-hero">
          Interactive Wave Field Solver
          <span class="badge">Prototype</span>
        </h1>
        <p class="lede">
          Set an incident wave condition and a neural surrogate model computes the
          three-dimensional flow field of nearshore wave breaking.
        </p>
        <ul class="tags">
          <li v-for="t in TAGS" :key="t" class="mono">{{ t }}</li>
        </ul>
      </div>

    </div>
  </div>

  <main class="inner">
    <div class="card">
      <nav class="tabs" role="tablist">
        <button
          v-for="t in TABS"
          :key="t.id"
          class="tab"
          :class="{ on: tab === t.id }"
          role="tab"
          :aria-selected="tab === t.id"
          @click="tab = t.id"
        >
          {{ t.label }}
        </button>
      </nav>

      <!-- v-show 而不是 v-if：切回来时视频不用重新加载，作业列表也留着 -->
      <div v-show="tab === 'demo'"><Experience /></div>
      <div v-show="tab === 'card'"><ModelCard /></div>
    </div>
  </main>

  <footer class="foot">
    <div class="inner t-small">
      <span>Oregon State University</span>
      <span class="dim">
        Surrogate model output, for research demonstration only — not a CFD solution
      </span>
    </div>
  </footer>
</template>

<style scoped>
.inner {
  max-width: var(--maxw);
  margin: 0 auto;
  padding: 0 1.5rem;
}

/* ---------- 顶栏 ---------- */
.topbar {
  border-bottom: 1px solid var(--line);
  background: var(--bg);
  position: sticky;
  top: 0;
  z-index: 10;
}
.topbar .inner {
  height: 56px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
.brand {
  display: flex;
  align-items: center;
  gap: 0.625rem;
  min-width: 0;
}
.mark {
  width: 32px;
  height: 20px;
  flex: none;
}
.name {
  font-size: 0.9375rem;
  font-weight: 500;
  white-space: nowrap;
}
.right {
  display: flex;
  align-items: center;
  gap: 1rem;
  min-width: 0;
}
.host {
  white-space: nowrap;
}
@media (max-width: 40rem) {
  .host {
    display: none;
  }
}

/* ---------- Hero ---------- */
.hero {
  padding: 2.5rem 0 2rem;
}
.htext {
  max-width: 44rem;
}
.eyebrow {
  color: var(--fg-3);
  margin-bottom: 0.5rem;
}
h1 {
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 0.625rem;
  margin-bottom: 0.625rem;
}
.badge {
  padding: 0.125rem 0.5rem;
  border: 1px solid var(--green);
  border-radius: var(--r-ctl);
  color: var(--green);
  font-size: 0.6875rem;
  font-weight: 400;
  letter-spacing: 0.02em;
  white-space: nowrap;
}
.lede {
  color: var(--fg-2);
  line-height: 1.65;
  margin-bottom: 1rem;
}
.tags {
  list-style: none;
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
}
.tags li {
  padding: 0.1875rem 0.625rem;
  border: 1px solid var(--green-dim);
  border-radius: 999px;
  color: var(--green);
  font-size: 0.75rem;
}

/* ---------- 主卡片 ----------
   比页面底色更深，是「凹」下去的内容区。32px 的大圆角配 4px 的方控件，
   这个反差是那套设计的识别点。 */
.card {
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: var(--r-card);
  padding: 1.75rem 0 0;
  overflow: hidden;
}

.tabs {
  display: flex;
  gap: 1.5rem;
  padding: 0 1.75rem;
  border-bottom: 1px solid var(--line);
  margin-bottom: 1.75rem;
}
.tab {
  position: relative;
  background: none;
  border: none;
  padding: 0 0 0.75rem;
  color: var(--fg-3);
  font: inherit;
  font-size: 0.9375rem;
  cursor: pointer;
  transition: color 0.12s ease;
}
.tab:hover {
  color: var(--fg-2);
}
.tab.on {
  color: var(--fg);
  font-weight: 500;
}
.tab.on::after {
  content: '';
  position: absolute;
  left: 0;
  right: 0;
  bottom: -1px;
  height: 2px;
  background: var(--green);
}

/* ---------- 页脚 ---------- */
.foot {
  margin-top: 3rem;
  padding: 1.5rem 0 3rem;
}
.foot .inner {
  display: flex;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  border-top: 1px solid var(--line);
  padding-top: 1.25rem;
}
.dim {
  color: var(--fg-3);
}
</style>
