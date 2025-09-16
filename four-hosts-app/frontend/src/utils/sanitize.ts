const HTML_TAG_PATTERN = /<[^>]+>/g
const WHITESPACE_PATTERN = /\s+/g
const DANGEROUS_CHARS_PATTERN = /[<>]/g

function toSafeString(value: unknown): string {
  if (value === null || value === undefined) return ''
  return String(value)
}

export function stripHtml(input: string): string {
  if (!input) return ''

  const normalized = toSafeString(input)
  const withoutTags = normalized
    .replace(HTML_TAG_PATTERN, ' ')
    .replace(DANGEROUS_CHARS_PATTERN, ' ')
  return withoutTags.replace(WHITESPACE_PATTERN, ' ').trim()
}

export function sanitize(input: string, max = 0): string {
  const s = stripHtml(input)
  if (max > 0 && s.length > max) return s.slice(0, max - 1) + '…'
  return s
}
