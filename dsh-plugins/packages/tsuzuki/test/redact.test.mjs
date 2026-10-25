import { test } from 'node:test'
import assert from 'node:assert/strict'
import { redact } from '../lib/redact.js'

const cases = [
  ['my key is sk-proj-abcdefghijklmnopqrstuvwxyz123456', 'my key is [REDACTED:OPENAI_KEY]', 'OPENAI_KEY'],
  ['use sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUVWXYZ0123', 'use [REDACTED:ANTHROPIC_KEY]', 'ANTHROPIC_KEY'],
  ['token ghp_abcdefghijklmnopqrstuvwxyz0123456789', 'token [REDACTED:GITHUB_TOKEN]', 'GITHUB_TOKEN'],
  ['hf_abcdefghijklmnopqrstuvwxyzABCDEF here', '[REDACTED:HF_TOKEN] here', 'HF_TOKEN'],
  ['AKIAIOSFODNN7EXAMPLE is aws', '[REDACTED:AWS_ACCESS_KEY] is aws', 'AWS_ACCESS_KEY'],
  ['key AIzaSyA-1234567890abcdefghijklmnopqrstu', 'key [REDACTED:GOOGLE_API_KEY]', 'GOOGLE_API_KEY'],
  ['Authorization: Bearer abcdefghijklmnopqrstuvwx.yz', 'Authorization: Bearer [REDACTED:BEARER]', 'BEARER'],
  ['postgres://admin:hunter2pass@db.local/x', 'postgres://admin:[REDACTED:URL_CREDENTIALS]@db.local/x', 'URL_CREDENTIALS'],
  ['password: hunter22', 'password: [REDACTED:SECRET_VALUE]', 'SECRET_VALUE'],
  ['API_KEY="abc123xyz"', 'API_KEY="[REDACTED:SECRET_VALUE]"', 'SECRET_VALUE'],
  ['パスワード：himitsu123です', 'パスワード：[REDACTED:SECRET_VALUE]です', 'SECRET_VALUE'],
  ['card 4111 1111 1111 1111 ok', 'card [REDACTED:CARD_NUMBER] ok', 'CARD_NUMBER'],
  ['-----BEGIN RSA PRIVATE KEY-----\nMIIabc\n-----END RSA PRIVATE KEY-----', '[REDACTED:PRIVATE_KEY]', 'PRIVATE_KEY'],
]

test('known secret shapes are redacted, labels kept', () => {
  for (const [input, expected, kind] of cases) {
    const r = redact(input)
    assert.equal(r.text, expected, input)
    assert.ok(r.found[kind] >= 1, kind)
  }
})

test('ordinary text is untouched', () => {
  for (const text of [
    'I moved to Kyoto last week and my cat Tama loves the garden.',
    '京都に引っ越した。電話は 090-1234-5678 です。',
    'The password policy requires 12 characters.',
    'Order #1234567890123 shipped; tracking 12345678901234.',
    'sk-short is not a key', 'Use the ask-anything mode',
    'function f(password) { return password.length }',
  ]) {
    const r = redact(text)
    assert.equal(r.text, text, text)
    assert.deepEqual(r.found, {})
  }
})

test('env-style names, more token formats and "my password is" are redacted (2026-10-25 review)', () => {
  for (const [input, expected] of [
    ['OPENAI_API_KEY=abcd1234efgh5678', 'OPENAI_API_KEY=[REDACTED:SECRET_VALUE]'],
    ['DB_PASSWORD=hunter2', 'DB_PASSWORD=[REDACTED:SECRET_VALUE]'],
    ['SECRET_KEY = "django-insecure-x1"', 'SECRET_KEY = "[REDACTED:SECRET_VALUE]"'],
    ['AWS_SECRET_ACCESS_KEY=wJalrXUtnFEMI/K7MDENG', 'AWS_SECRET_ACCESS_KEY=[REDACTED:SECRET_VALUE]'],
    ['token: 9f8e7d6c5b4a', 'token: [REDACTED:SECRET_VALUE]'],
    ['Authorization: Basic dXNlcjpwYXNzd29yZDEyMw==', 'Authorization: Basic [REDACTED:BASIC_AUTH]'],
    ['glpat-abcdefghijklmnopqrstu', '[REDACTED:GITLAB_TOKEN]'],
    ['npm_abcdefghijklmnopqrstuvwxyz0123456789', '[REDACTED:NPM_TOKEN]'],
    ['whsec_abcdefghijklmnopqrstuv', '[REDACTED:STRIPE_WEBHOOK_SECRET]'],
    ['https://hooks.slack.com/services/T000/B000/XXXX', '[REDACTED:SLACK_WEBHOOK]'],
    ['my password is hunter2', 'my password is [REDACTED:SECRET_VALUE]'],
    ['パスワードは abc123 です', 'パスワードは [REDACTED:SECRET_VALUE] です'],
  ]) assert.equal(redact(input).text, expected, input)
  for (const text of ['the password is required', 'pwd: /Users/me/project', 'set max_tokens = 1024', 'The token count is high.'])
    assert.equal(redact(text).text, text, text)
})

test('redaction stays linear on long adversarial input (2026-10-25 review)', () => {
  for (const text of ['password' + ' '.repeat(120_000), 'password' + '\t'.repeat(120_000), 'de-'.repeat(40_000), 'a_'.repeat(60_000)]) {
    const t0 = performance.now()
    redact(text)
    assert.ok(performance.now() - t0 < 500, `took ${(performance.now() - t0).toFixed(0)} ms`)
  }
})
