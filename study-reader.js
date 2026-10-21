'use strict';
// Reader chrome only: no quiz/storage mutations.
(() => {
  const menu = document.querySelector('.reader-menu');
  const toc = document.getElementById('reader-toc');
  const backdrop = document.querySelector('.reader-backdrop');
  if (!menu || !toc || !backdrop) return;
  document.body.classList.add('reader-ready');
  const mobile = window.matchMedia('(max-width:800px)');
  const setOpen = (open, restoreFocus = false) => {
    document.body.classList.toggle('reader-toc-open', open);
    menu.setAttribute('aria-expanded', String(open));
    backdrop.hidden = !open;
    if (open) {
      const contents = toc.querySelector('details');
      if (contents) contents.open = true;
      toc.querySelector('a')?.focus({preventScroll:true});
    } else if (restoreFocus) menu.focus();
  };
  menu.addEventListener('click', () => setOpen(menu.getAttribute('aria-expanded') !== 'true'));
  backdrop.addEventListener('click', () => setOpen(false, true));
  document.addEventListener('keydown', event => {
    if (!document.body.classList.contains('reader-toc-open')) return;
    if (event.key === 'Escape') setOpen(false, true);
    if (event.key === 'Tab') {
      const nodes = [menu, ...toc.querySelectorAll('a, summary')].filter(el => el.getClientRects().length);
      const first = nodes[0], last = nodes[nodes.length - 1];
      if (event.shiftKey && document.activeElement === first) { event.preventDefault(); last.focus(); }
      if (!event.shiftKey && document.activeElement === last) { event.preventDefault(); first.focus(); }
    }
  });
  toc.addEventListener('click', event => {
    const link = event.target.closest('a');
    if (link && mobile.matches) {
      setOpen(false);
      if (link.hash && link.pathname === location.pathname) {
        const target = document.getElementById(decodeURIComponent(link.hash.slice(1)));
        if (target) { target.setAttribute('tabindex', '-1'); target.focus({preventScroll:true}); }
      }
    }
  });
  mobile.addEventListener('change', () => {
    setOpen(false);
    if (!mobile.matches) { const contents = toc.querySelector('details'); if (contents) contents.open = true; }
  });
  const links = [...toc.querySelectorAll('a[href^="#"]')];
  const sections = links.map(link => [link, document.getElementById(decodeURIComponent(link.hash.slice(1)))]).filter(([,el]) => el);
  let ticking = false;
  const update = () => {
    const distance = document.documentElement.scrollHeight - innerHeight;
    document.documentElement.style.setProperty('--reading-progress', `${distance > 0 ? Math.min(100, scrollY / distance * 100) : 0}%`);
    let current = sections.find(([,el]) => el.getClientRects().length)?.[0];
    let nearest = -Infinity;
    for (const [link, el] of sections) {
      const top = el.getBoundingClientRect().top;
      if (el.getClientRects().length && top <= 140 && top > nearest) { current = link; nearest = top; }
    }
    for (const link of links) { link.classList.toggle('is-current', link === current); if (link === current) link.setAttribute('aria-current','location'); else link.removeAttribute('aria-current'); }
    ticking = false;
  };
  addEventListener('scroll', () => { if (!ticking) { ticking = true; requestAnimationFrame(update); } }, {passive:true});
  update();
})();
