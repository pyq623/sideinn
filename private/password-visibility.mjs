for (const toggle of document.querySelectorAll('[data-password-visibility]')) {
  const fields = toggle.dataset.passwordVisibility.split(' ').map(id => document.getElementById(id));
  toggle.addEventListener('change', () => {
    for (const field of fields) field.type = toggle.checked ? 'text' : 'password';
  });
  toggle.disabled = false;
}
