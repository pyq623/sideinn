(()=>{
 'use strict';
 for(const button of document.querySelectorAll('[data-copy]')){
  button.addEventListener('click',async()=>{
   const code=document.getElementById(button.dataset.copy);
   const status=button.parentElement.querySelector('[role="status"]');
   try{await navigator.clipboard.writeText(code.textContent);status.textContent='已复制完整代码';}
   catch{const selection=window.getSelection(),range=document.createRange();range.selectNodeContents(code);selection.removeAllRanges();selection.addRange(range);status.textContent='代码已选中，请按 Ctrl/Cmd+C 复制';}
  });
 }
})();
