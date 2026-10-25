/**
 * Secret redaction before anything reaches the memory store, the embedding provider or the fact model.
 * Each match becomes `[REDACTED:KIND]`; ordinary text is left untouched. Patterns favour precision: a
 * missed exotic secret is possible, a mangled sentence is not acceptable.
 */

interface Rule {
  kind: string
  pattern: RegExp
  /** Keep group 1 (e.g. the `password:` label) and redact only the value. */
  keepLabel?: boolean
}

const RULES: Rule[] = [
  { kind: 'PRIVATE_KEY', pattern: /-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----/g },
  { kind: 'ANTHROPIC_KEY', pattern: /\bsk-ant-[A-Za-z0-9_-]{20,}/g },
  { kind: 'OPENAI_KEY', pattern: /\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_-]{20,}/g },
  { kind: 'GITHUB_TOKEN', pattern: /\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,})/g },
  { kind: 'HF_TOKEN', pattern: /\bhf_[A-Za-z0-9]{30,}/g },
  { kind: 'AWS_ACCESS_KEY', pattern: /\b(?:AKIA|ASIA)[A-Z0-9]{16}\b/g },
  { kind: 'GOOGLE_API_KEY', pattern: /\bAIza[0-9A-Za-z_-]{35}\b/g },
  { kind: 'SLACK_TOKEN', pattern: /\bxox[abprs]-[A-Za-z0-9-]{10,}/g },
  { kind: 'STRIPE_KEY', pattern: /\b(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{20,}/g },
  { kind: 'JWT', pattern: /\beyJ[A-Za-z0-9_-]{10,}\.eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}/g },
  { kind: 'BEARER', pattern: /\b(Bearer\s+)[A-Za-z0-9._~+/-]{20,}=*/g, keepLabel: true },
  { kind: 'URL_CREDENTIALS', pattern: /\b([a-z][a-z0-9+.-]{0,31}:\/\/[^\s:/@]+:)[^\s@/]+(?=@)/gi, keepLabel: true },
  { kind: 'BASIC_AUTH', pattern: /\b(Basic\s+)[A-Za-z0-9+/]{16,}={0,2}/g, keepLabel: true },
  { kind: 'GITLAB_TOKEN', pattern: /\bglpat-[A-Za-z0-9_-]{20,}/g },
  { kind: 'NPM_TOKEN', pattern: /\bnpm_[A-Za-z0-9]{36}\b/g },
  { kind: 'SENDGRID_KEY', pattern: /\bSG\.[A-Za-z0-9_-]{16,}\.[A-Za-z0-9_-]{16,}/g },
  { kind: 'STRIPE_WEBHOOK_SECRET', pattern: /\bwhsec_[A-Za-z0-9]{20,}/g },
  { kind: 'SLACK_WEBHOOK', pattern: /https:\/\/hooks\.slack\.com\/services\/[A-Za-z0-9/_-]+/g },
  { kind: 'DISCORD_TOKEN', pattern: /\b[MNO][A-Za-z\d_-]{23,27}\.[A-Za-z\d_-]{6}\.[A-Za-z\d_-]{27,40}\b/g },
  // `password: x`, `OPENAI_API_KEY=x`, `パスワード：x`, `secret="x"`: the label stays, the value goes. A label may carry
  // a prefix (`DB_PASSWORD`, `AWS_SECRET_ACCESS_KEY`). Values are ASCII-printable runs, so Japanese text written right
  // after a value (`…123です`) is kept.
  { kind: 'SECRET_VALUE',
    pattern: /((?:(?<![A-Za-z0-9])(?:[A-Za-z0-9]{1,24}[_-]){0,4}(?:password|passwd|passphrase|secret(?:[_-]?(?:access[_-]?)?key)?|api[_-]?key|access[_-]?token|auth[_-]?token|token|client[_-]?secret|private[_-]?key)(?![A-Za-z0-9])|パスワード|暗証番号|シークレット|APIキー)\s*(?:["']\s*)?[:=：]\s*["']?)((?:(?!["'`,;])[\x21-\x7e]){4,})/gi,
    keepLabel: true },
  // "my password is hunter2", "パスワードは abc123": only values with a digit or a symbol, so "the password is
  // required" stays as written.
  { kind: 'SECRET_VALUE',
    pattern: /((?<![A-Za-z0-9])(?:password|passcode|passphrase)\s+is\s+|(?:パスワード|暗証番号)は\s*)((?=[\x21-\x7e]*[0-9!@#$%^&*_+=~-])[\x21-\x7e]{4,})/gi,
    keepLabel: true },
]

/** Luhn-valid 13–19 digit card numbers (spaces or dashes allowed). */
const CARD = /\b\d(?:[ -]?\d){12,18}\b/g

function luhn(digits: string): boolean {
  let sum = 0
  for (let i = 0; i < digits.length; i++) {
    let d = Number(digits[digits.length - 1 - i])
    if (i % 2 === 1) { d *= 2; if (d > 9) d -= 9 }
    sum += d
  }
  return sum % 10 === 0
}

export interface Redaction {
  text: string
  /** Count per kind; empty when nothing was redacted. */
  found: Record<string, number>
}

export function redact(text: string): Redaction {
  const found: Record<string, number> = {}
  let out = text
  for (const rule of RULES) {
    out = out.replace(rule.pattern, (...m: string[]) => {
      found[rule.kind] = (found[rule.kind] ?? 0) + 1
      return rule.keepLabel ? `${m[1]}[REDACTED:${rule.kind}]` : `[REDACTED:${rule.kind}]`
    })
  }
  out = out.replace(CARD, match => {
    const digits = match.replace(/[ -]/g, '')
    if (digits.length < 13 || digits.length > 19 || !luhn(digits)) return match
    found.CARD_NUMBER = (found.CARD_NUMBER ?? 0) + 1
    return '[REDACTED:CARD_NUMBER]'
  })
  return { text: out, found }
}
