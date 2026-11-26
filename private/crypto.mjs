const saltBytes = 16;
const ivBytes = 12;
const archiveBytes = 1024 * 1024;
const iterations = 600000;
const encoder = new TextEncoder();
const decoder = new TextDecoder('utf-8', { fatal: true });

async function deriveKey(password, salt, use) {
  const passwordBytes = encoder.encode(password);
  try {
    const material = await crypto.subtle.importKey('raw', passwordBytes, 'PBKDF2', false, ['deriveKey']);
    return await crypto.subtle.deriveKey(
      { name: 'PBKDF2', salt, iterations, hash: 'SHA-256' },
      material,
      { name: 'AES-GCM', length: 256 },
      false,
      [use]
    );
  } finally {
    passwordBytes.fill(0);
  }
}

export async function seal(password, data) {
  const packed = encoder.encode(JSON.stringify(data));
  if (packed.length > archiveBytes - 4) throw new Error('The archive exceeds its fixed capacity.');
  const plain = new Uint8Array(archiveBytes);
  new DataView(plain.buffer).setUint32(0, packed.length);
  plain.set(packed, 4);
  packed.fill(0);
  const salt = crypto.getRandomValues(new Uint8Array(saltBytes));
  const iv = crypto.getRandomValues(new Uint8Array(ivBytes));
  try {
    const key = await deriveKey(password, salt, 'encrypt');
    const encrypted = new Uint8Array(await crypto.subtle.encrypt({ name: 'AES-GCM', iv }, key, plain));
    const file = new Uint8Array(saltBytes + ivBytes + encrypted.length);
    file.set(salt, 0);
    file.set(iv, saltBytes);
    file.set(encrypted, saltBytes + ivBytes);
    return file;
  } finally {
    plain.fill(0);
  }
}

export async function openArchive(password, file) {
  if (file.byteLength !== saltBytes + ivBytes + archiveBytes + 16) throw new Error('Invalid archive length.');
  const salt = file.slice(0, saltBytes);
  const iv = file.slice(saltBytes, saltBytes + ivBytes);
  const key = await deriveKey(password, salt, 'decrypt');
  const plain = new Uint8Array(await crypto.subtle.decrypt(
    { name: 'AES-GCM', iv }, key, file.slice(saltBytes + ivBytes)
  ));
  try {
    const length = new DataView(plain.buffer).getUint32(0);
    if (length > plain.length - 4) throw new Error('Invalid archive contents.');
    const data = JSON.parse(decoder.decode(plain.subarray(4, 4 + length)));
    if (data.version !== 1 || !Array.isArray(data.notes)) throw new Error('Unsupported archive.');
    return data.notes;
  } finally {
    plain.fill(0);
  }
}
