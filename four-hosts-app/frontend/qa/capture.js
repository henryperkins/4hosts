// Simple QA capture script using Playwright (Chromium)
// Produces before.png and after.png and an optional GIF if ffmpeg is available.

const { chromium } = require('playwright')
const fs = require('fs')
const { execSync } = require('child_process')

async function main() {
  const base = process.env.APP_URL || 'http://localhost:5173'
  const query = process.env.Q || 'Tell me about context engineering and LLM hallucinations'
  const outDir = process.env.OUT || 'qa-out'
  fs.mkdirSync(outDir, { recursive: true })

  const browser = await chromium.launch({ headless: true })
  const context = await browser.newContext({ viewport: { width: 390, height: 844 } }) // iPhone-ish
  const page = await context.newPage()

  // BEFORE: home with query typed
  await page.goto(base, { waitUntil: 'domcontentloaded' })
  // type query if the textarea exists
  const ta = await page.$('textarea, input[type="text"]')
  if (ta) {
    await ta.fill(query)
  }
  await page.screenshot({ path: `${outDir}/before.png`, fullPage: true })

  // Submit if there is a button labelled “Ask the Four Hosts”
  const btn = await page.$('button:has-text("Ask the Four Hosts")')
  if (btn) {
    await btn.click()
  }

  // Wait a bit for progress to render
  await page.waitForTimeout(2500)
  await page.screenshot({ path: `${outDir}/during.png`, fullPage: true })

  // Try to expand Recent Sources panel if present
  const recent = await page.$('text=Recent Sources')
  if (recent) {
    await recent.click()
    await page.waitForTimeout(500)
  }

  await page.screenshot({ path: `${outDir}/after.png`, fullPage: true })

  await browser.close()

  // Optional GIF assembly if ffmpeg present
  try {
    execSync(`ffmpeg -y -i ${outDir}/during.png -i ${outDir}/after.png -filter_complex "[0][1]concat=n=2:v=1[out]" -map "[out]" ${outDir}/progress.gif`, { stdio: 'ignore' })
    console.log(`GIF saved to ${outDir}/progress.gif`)
  } catch (e) {
    console.log('ffmpeg not available; skipped GIF assembly')
  }
}

main().catch((e) => {
  console.error(e)
  process.exit(1)
})

