import { onMounted, onUnmounted } from 'vue'

/**
 * 把「真人还在用」这件事告诉后端。
 *
 * 后端闲置一段时间会自动停止（它跑在共享的登录节点上，不留没人管的进程），
 * 而**后台轮询不算操作** —— 否则一个忘了关的标签页就能把服务永远吊着。
 * 所以真人的点击/按键要单独报一次。
 *
 * 节流到每分钟最多一次：闲置阈值是小时级的，报得再密也没有意义，
 * 只会在页面滑动时刷出一堆没用的请求。
 */
const THROTTLE_MS = 60_000

export function useActivity() {
  let last = 0

  function ping() {
    const t = Date.now()
    if (t - last < THROTTLE_MS) return
    last = t
    // 失败无所谓：最坏结果是后端早一点自停，重开就是了
    fetch('/api/active', { method: 'POST' }).catch(() => {})
  }

  function onVisible() {
    if (document.visibilityState === 'visible') ping()
  }

  onMounted(() => {
    // pointerdown / keydown 才是真人；mousemove 太吵，滚动也可能是惯性
    window.addEventListener('pointerdown', ping, { passive: true })
    window.addEventListener('keydown', ping, { passive: true })
    document.addEventListener('visibilitychange', onVisible)
  })

  onUnmounted(() => {
    window.removeEventListener('pointerdown', ping)
    window.removeEventListener('keydown', ping)
    document.removeEventListener('visibilitychange', onVisible)
  })
}
