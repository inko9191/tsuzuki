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
