(() => {
  'use strict';
  const root = document.querySelector('[data-coding-course]');
  if (!root) return;
  const $ = id => document.getElementById(id);
  const cards = [...root.querySelectorAll('.coding-question')];
  const suffix = new URLSearchParams(location.search).has('preview') ? ':preview' : '';
  let records = {}, writable = true, key = 'sideinn:coding:' + root.dataset.codingCourse + ':v1' + suffix;
  try {
    ({key, records, writable} = CodingState.load(localStorage, root.dataset.codingCourse, suffix, cards.map(card => card.id)));
  } catch (_) { writable = false; }
  // Keep failed edits until a later write succeeds; only those fields may
  // override the latest snapshot written by another open practice tab.
  const pending = writable ? {} : JSON.parse(JSON.stringify(records));
  function save(id, field) {
    pending[id] ||= {};
    pending[id][field] = records[id][field];
    pending[id].updated = records[id].updated = new Date().toISOString();
    try {
      let latest;
      try { latest = JSON.parse(localStorage.getItem(key)); }
      catch (error) { if (error instanceof SyntaxError) latest = {}; else throw error; }
      const valid = value => value && typeof value === 'object' && !Array.isArray(value);
      if (!valid(latest)) latest = {};
      const merged = {};
      for (const card of cards) {
        merged[card.id] = {...(valid(latest[card.id]) ? latest[card.id] : records[card.id]), ...pending[card.id]};
        if (!['todo', 'review', 'done'].includes(merged[card.id].status)) merged[card.id].status = 'todo';
        if (!Array.isArray(merged[card.id].checks)) merged[card.id].checks = [];
        for (const name of ['draft', 'notes']) {
          if (typeof merged[card.id][name] !== 'string') merged[card.id][name] = '';
        }
      }
      localStorage.setItem(key, JSON.stringify(merged));
      for (const card of cards) {
        Object.assign(records[card.id], merged[card.id]);
        const record = records[card.id];
        for (const name of ['draft', 'notes']) {
          const area = card.querySelector('.coding-' + name);
          const value = typeof record[name] === 'string' ? record[name] : '';
          if (area.value !== value) area.value = value;
        }
        card.querySelector('[data-review]').value = record.status || 'todo';
        card.querySelectorAll('[data-check]').forEach(box => {box.checked = (record.checks || []).includes(box.dataset.check);});
      }
      for (const id of Object.keys(pending)) delete pending[id];
      render();
      writable = true;
      $('coding-save').textContent = '已保存到当前浏览器。代码不会在页面内执行；可导出记录备份。';
    } catch (_) {
      writable = false;
      $('coding-save').textContent = '浏览器存储不可用或已满，本次内容暂存内存；离开前请导出记录。';
    }
  }
  function render() {
    const query = $('coding-search').value.trim().toLowerCase();
    const tier = $('coding-tier').value, status = $('coding-filter').value;
    let visible = 0, done = 0;
    for (const card of cards) {
      const state = records[card.id]?.status || 'todo';
      if (state === 'done') done++;
      card.hidden = !!((tier && card.dataset.tier !== tier) || (status && state !== status) || (query && !card.textContent.toLowerCase().includes(query)));
      if (!card.hidden) visible++;
    }
    $('coding-progress').textContent = '已掌握 ' + done + ' / ' + cards.length + ' 道 · 当前显示 ' + visible + ' 道';
    $('coding-meter').value = done;
    $('coding-empty').hidden = visible !== 0;
  }
  for (const card of cards) {
    let record = records[card.id];
    if (!record || typeof record !== 'object' || Array.isArray(record)) record = {};
    records[card.id] = record;
    if (!['todo','review','done'].includes(record.status)) record.status = 'todo';
    record.checks = Array.isArray(record.checks) ? record.checks : [];
    card.querySelector('[data-review]').value = record.status;
    for (const field of ['draft','notes']) {
      const area = card.querySelector('.coding-' + field);
      area.value = typeof record[field] === 'string' ? record[field] : '';
      area.addEventListener('input', () => {record[field] = area.value; save(card.id, field);});
    }
    card.querySelectorAll('[data-check]').forEach(box => {
      box.checked = record.checks.includes(box.dataset.check);
      box.addEventListener('change', () => {
        record.checks = [...card.querySelectorAll('[data-check]:checked')].map(b => b.dataset.check); save(card.id, 'checks');
      });
    });
    card.querySelector('[data-review]').addEventListener('change', e => {record.status=e.target.value; save(card.id, 'status'); render();});
  }
  for (const id of ['coding-search','coding-tier','coding-filter']) $(id).addEventListener('input',render);
  root.addEventListener('click', async e => {
    const button = e.target.closest('[data-copy]');
    if (!button) return;
    try {await navigator.clipboard.writeText($(button.dataset.copy).textContent); button.textContent='已复制';}
    catch (_) {button.textContent='复制不可用，请选中代码复制';}
  });
  $('coding-export').addEventListener('click', () => {
    const blob = new Blob([JSON.stringify({version:1,course:root.dataset.codingCourse,exported:new Date().toISOString(),records},null,2)],{type:'application/json'});
    const url=URL.createObjectURL(blob), a=document.createElement('a');
    a.href=url; a.download='ai-infra-review.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  function revealAnchor() {
    let id;
    try { id = decodeURIComponent(location.hash.slice(1)); }
    catch (_) { return; }
    const target=document.getElementById(id), card=target?.closest('.coding-question');
    if (card) {
      if(card.hidden){$('coding-search').value='';$('coding-tier').value='';$('coding-filter').value='';render();}
      card.scrollIntoView({block:'start'});
    }
  }
  window.addEventListener('hashchange',revealAnchor);
  if (!writable) $('coding-save').textContent='暂未读取到有效存储；可继续练习，建议导出记录备份。';
  render(); revealAnchor();
})();
