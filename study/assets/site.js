'use strict';
// Reading navigation remains usable when scripts or storage are unavailable.
if (window.matchMedia('(max-width: 800px)').matches) {
  const toc = document.querySelector('.course-toc > details');
  if (toc) toc.open = false;
}
const progressData = document.getElementById('progress-data');
if (progressData) for (const config of JSON.parse(progressData.textContent)) {
  const el = document.getElementById('progress-' + config.unit);
  try {
    const saved = StudyRecords.read(localStorage, config, config.question_ids);
    if (!saved.ok) throw new Error('Storage unavailable');
    const records = Object.values(saved.records).filter(x => x.submitted === true);
    el.textContent = '本轮已提交 ' + records.length + ' · 答对 ' + records.filter(x => x.last === true).length;
  } catch (_) { el.textContent = '当前无法读取记录，仍可阅读和练习。'; }
}
const searchInput = document.getElementById('search');
if (searchInput) {
  const entries = JSON.parse(document.getElementById('search-data').textContent);
  const list = document.getElementById('search-results');
  const status = document.getElementById('search-status');
  searchInput.addEventListener('input', () => {
    list.replaceChildren();
    const terms = searchInput.value.trim().toLocaleLowerCase().split(/\s+/).filter(Boolean);
    if (!terms.length) { status.textContent = ''; return; }
    const matches = entries.filter(item => terms.every(term => (item.title + ' ' + item.course).toLocaleLowerCase().includes(term)));
    status.textContent = matches.length ? '找到 ' + matches.length + ' 个章节' : '未找到匹配章节，可以换一个更短的关键词。';
    for (const item of matches) {
      const li = document.createElement('li');
      const link = document.createElement('a');
      link.href = item.url;
      link.textContent = item.course + ' · ' + item.title;
      li.append(link); list.append(li);
    }
  });
}
