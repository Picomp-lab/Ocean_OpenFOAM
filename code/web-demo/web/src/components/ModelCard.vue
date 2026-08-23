<script setup lang="ts">
/**
 * 「模型说明」页 —— 只讲链路：一次提交在集群上依次经过哪三步。
 *
 * 耗时是集群上的实测值，不是估的。改数字前先回去对 `models/README.md`。
 */
</script>

<template>
  <div class="card-body">
    <h2 class="t-section">Pipeline</h2>
    <ol class="pipe">
      <li>
        <span class="n">1</span>
        <div>
          <b>FUNWAVE-TVD</b><span class="tag">Boussinesq</span>
          <p>
            Depth-averaged 2D solver giving (η, u, v) at time t. Measured on a single core:
            <span class="tnum">15,795 s</span> for 100 s of physical time. Under the default
            configuration this stage reuses output already on the cluster rather than re-running.
          </p>
        </div>
      </li>
      <li>
        <span class="n">2</span>
        <div>
          <b>Nwogu lift</b><span class="tag">No free parameters</span>
          <p>
            Lifts the 2D state into a 3D field using the quadratic profile of Nwogu (1993).
            The profile is analytic in z, so nothing is interpolated in the vertical; dry cells
            are set to NaN rather than extrapolated. About 6 min.
          </p>
        </div>
      </li>
      <li>
        <span class="n">3</span>
        <div>
          <b>HPM</b><span class="tag">740k parameters</span>
          <p>
            Holistic Physics Mixer (ICML 2025). LBO eigenvectors of the OpenFOAM graph
            Laplacian form a fixed spectral basis (first 64 modes); a per-point gate predicts
            frequency preference, mixing happens in the spectral domain, and the network
            outputs only the residual from prior to ground truth. A 1000-frame rollout plus
            four-channel rendering takes about 1 h 38 min.
          </p>
        </div>
      </li>
    </ol>
  </div>
</template>

<style scoped>
.card-body {
  padding: 0 1.75rem 2rem;
  max-width: 56rem;
}
h2 {
  margin-bottom: 0.875rem;
}
b {
  color: var(--fg);
  font-weight: 500;
}

.pipe {
  list-style: none;
  display: grid;
  gap: 1px;
  background: var(--line);
  border: 1px solid var(--line);
  border-radius: var(--r-panel);
  overflow: hidden;
}
.pipe li {
  display: flex;
  gap: 0.875rem;
  padding: 1rem 1.125rem;
  background: var(--surface-2);
}
.pipe .n {
  flex: none;
  width: 1.5rem;
  height: 1.5rem;
  border-radius: 50%;
  border: 1px solid var(--green);
  color: var(--green);
  font-size: 0.75rem;
  display: flex;
  align-items: center;
  justify-content: center;
}
.pipe p {
  margin: 0.3125rem 0 0;
  font-size: 0.875rem;
  color: var(--fg-2);
  line-height: 1.7;
}
.tag {
  display: inline-block;
  margin-left: 0.5rem;
  padding: 0.0625rem 0.4375rem;
  border: 1px solid var(--line-strong);
  border-radius: var(--r-ctl);
  font-size: 0.6875rem;
  color: var(--fg-3);
}
</style>
