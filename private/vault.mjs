import { openArchive } from './crypto.mjs';

const byId = id => document.getElementById(id);
const form = byId('unlock');
const password = byId('password');
const status = byId('status');
const content = byId('content');
const download = byId('download');
const lockButton = byId('lock');
const back = byId('back');
let notes = [];
let encrypted;
let downloadURL = '';
let privateStyles;
let lockVersion = 0;

function clearReader() {
  content.replaceChildren();
  privateStyles?.remove();
  privateStyles = undefined;
  if (downloadURL) URL.revokeObjectURL(downloadURL);
  downloadURL = '';
  download.removeAttribute('href');
  download.removeAttribute('download');
  download.hidden = true;
  back.hidden = true;
}

function showNote(note) {
  clearReader();
  const fragment = document.createElement('template');
  fragment.innerHTML = note.html;
  fragment.content.querySelectorAll('script, form, iframe, object, embed, link, meta, base').forEach(element => element.remove());
  for (const element of fragment.content.querySelectorAll('*')) {
    for (const { name, value } of [...element.attributes]) {
      if (name.startsWith('on') || name === 'srcdoc') element.removeAttribute(name);
      if (name === 'href' && !/^(https?:\/\/|#)/i.test(value)) element.removeAttribute(name);
    }
    if (element.matches('a[href^="http"]')) {
      element.target = '_blank';
      element.rel = 'noopener noreferrer';
    }
  }
  if (note.css) {
    privateStyles = document.createElement('style');
    privateStyles.textContent = note.css;
    document.head.append(privateStyles);
  }
  const article = document.createElement('article');
  article.className = 'vault-article';
  article.append(fragment.content);
  content.append(article);
  downloadURL = URL.createObjectURL(new Blob([note.markdown], { type: 'text/markdown;charset=utf-8' }));
  download.href = downloadURL;
  download.download = note.title + '.md';
  download.hidden = false;
  back.hidden = false;
  window.scrollTo(0, 0);
  back.focus();
}

function showDirectory() {
  clearReader();
  const heading = document.createElement('h1');
  heading.textContent = '私密笔记';
  const list = document.createElement('ul');
  list.className = 'vault-notes';
  for (const note of notes) {
    const row = document.createElement('li');
    const button = document.createElement('button');
    button.textContent = note.title;
    button.addEventListener('click', () => showNote(note));
    row.append(button);
    list.append(row);
  }
  content.append(heading, list);
  window.scrollTo(0, 0);
}

function lock() {
  lockVersion += 1;
  notes = [];
  clearReader();
  form.reset();
  password.type = 'password';
  status.textContent = '';
  byId('unlocked').hidden = true;
  lockButton.hidden = true;
  byId('locked').hidden = false;
}

lockButton.addEventListener('click', () => { lock(); password.focus(); });
back.addEventListener('click', showDirectory);
content.addEventListener('click', event => {
  const link = event.target.closest('a[href^="#"]');
  if (!link) return;
  event.preventDefault();
  const target = document.getElementById(link.getAttribute('href').slice(1));
  if (content.contains(target)) target.scrollIntoView();
});
window.addEventListener('pagehide', lock);
form.addEventListener('submit', async event => {
  event.preventDefault();
  const button = byId('unlock-button');
  const input = password.value;
  const version = lockVersion;
  password.value = '';
  button.disabled = true;
  status.textContent = '正在解锁…';
  try {
    if (!encrypted) {
      const response = await fetch('./archive.bin', { cache: 'no-store' });
      if (!response.ok) throw new Error('Archive unavailable.');
      encrypted = await response.arrayBuffer();
    }
    const decrypted = await openArchive(input, encrypted);
    if (version !== lockVersion) return;
    notes = decrypted;
    showDirectory();
    status.textContent = '';
    byId('locked').hidden = true;
    byId('unlocked').hidden = false;
    lockButton.hidden = false;
    lockButton.focus();
  } catch {
    if (version === lockVersion) {
      status.textContent = '未能解锁，请确认密码后重试。';
      password.focus();
    }
  } finally {
    button.disabled = false;
  }
});
byId('unlock-button').disabled = false;
