'use strict';
(function () {
  const mapping = JSON.parse(document.getElementById('legacy-links').textContent);
  function followMovedSection() {
    let anchor;
    try { anchor = decodeURIComponent(location.hash.slice(1)); } catch (_) { return; }
    if (!Object.prototype.hasOwnProperty.call(mapping, anchor)) return;
    const target = new URL(mapping[anchor], location.href);
    target.search = location.search;
    location.replace(target.href);
  }
  window.addEventListener('hashchange', followMovedSection);
  followMovedSection();
})();
