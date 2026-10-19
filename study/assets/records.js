'use strict';
// Read old question records without moving/deleting the original storage entries.
// A destination record, including a reset/draft, always overrides its legacy copy.
(function (root) {
  function load(storage, key) {
    const data = JSON.parse(storage.getItem(key) || '{}');
    if (!data || typeof data !== 'object' || Array.isArray(data)) throw new Error('Invalid records');
    return data;
  }
  function read(storage, config, ids, suffix = '') {
    let current = {}, legacy = {}, ok = true;
    try { current = load(storage, config.storage_key + suffix); } catch (_) { ok = false; }
    if (config.legacy_storage_key) {
      try { legacy = load(storage, config.legacy_storage_key + suffix); } catch (_) { ok = false; }
    }
    const records = {};
    for (const id of ids) {
      const record = Object.prototype.hasOwnProperty.call(current, id) ? current[id] : legacy[id];
      if (record && typeof record === 'object' && !Array.isArray(record)) records[id] = record;
    }
    return { records, ok };
  }
  function write(storage, key, records) {
    try {
      // Preserve unrelated records if a previous version shared this storage key.
      const existing = load(storage, key);
      storage.setItem(key, JSON.stringify({ ...existing, ...records }));
      return true;
    } catch (_) { return false; }
  }
  root.StudyRecords = { read, write };
})(globalThis);
