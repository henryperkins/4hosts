export function stripHtml(input: string): string {
  if (!input) return ''
  try {
    const txt = input.replace(/<[^>]+>/g, ' ')
    return txt.replace(/\s+/g, ' ').trim()
  } catch {
    return input
  }
}

export function sanitize(input: string, max = 0): string {
  const s = stripHtml(input)
  if (max > 0 && s.length > max) return s.slice(0, max - 1) + '…'
  return s
}

