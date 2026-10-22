(function () {
  'use strict';

  // Delegation also covers content inserted by manual or remembered decryption.
  if (window.jpAddressCopyInstalled) return;
  window.jpAddressCopyInstalled = true;
  var resets = new WeakMap();

  function legacyCopy(text) {
    var active = document.activeElement;
    var selection = window.getSelection();
    var ranges = [];
    for (var i = 0; selection && i < selection.rangeCount; i++) ranges.push(selection.getRangeAt(i).cloneRange());
    var input = document.createElement('textarea');
    input.value = text;
    input.setAttribute('aria-label', '待复制的资料');
    input.style.cssText = 'position:fixed;top:0;left:-9999px;font-size:16px;';
    // Avoid the theme appending a copyright notice to the exact field value.
    input.addEventListener('copy', function (event) { event.stopPropagation(); });
    document.body.appendChild(input);
    try {
      input.focus({ preventScroll: true });
      input.select();
      if (!document.execCommand('copy')) throw new Error('Copy unavailable');
    } finally {
      input.remove();
      if (active && active.isConnected) active.focus({ preventScroll: true });
      if (selection) {
        selection.removeAllRanges();
        ranges.forEach(function (range) { selection.addRange(range); });
      }
    }
  }

  async function copyText(text) {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === 'function') {
      try {
        await navigator.clipboard.writeText(text);
        return;
      } catch (_) { /* Try the compatibility path when permission is denied. */ }
    }
    legacyCopy(text);
  }

  document.addEventListener('click', async function (event) {
    if (!(event.target instanceof Element)) return;
    var button = event.target.closest('button[data-jp-copy]');
    var card = button && button.closest('.jp-address-card');
    if (!card || button.disabled) return;
    var all = button.dataset.jpCopy === 'all';
    var field = all ? null : button.closest('.jp-address-field');
    var fields = all ? Array.from(card.querySelectorAll('.jp-address-field')) : [field];
    var text = fields.map(function (item) {
      var value = item.querySelector('.jp-address-value').textContent.trim();
      return all ? item.querySelector('.jp-address-label').textContent.trim() + '：' + value : value;
    }).join('\n');
    var label = button.querySelector('[data-jp-button-text]');
    var status = card.querySelector('.jp-address-status');
    var restoreFocus = document.activeElement === button;
    window.clearTimeout(resets.get(button));
    button.disabled = true;
    label.textContent = '复制中';
    status.textContent = '';
    status.removeAttribute('data-error');
    try {
      await copyText(text);
      button.setAttribute('data-copied', '');
      label.textContent = '已复制';
      status.textContent = all ? '全部资料已复制。' : field.querySelector('.jp-address-label').textContent.trim() + '已复制。';
      resets.set(button, window.setTimeout(function () {
        label.textContent = all ? '复制全部' : '复制';
        button.removeAttribute('data-copied');
      }, 1800));
    } catch (_) {
      button.removeAttribute('data-copied');
      label.textContent = all ? '复制全部' : '复制';
      status.setAttribute('data-error', '');
      status.textContent = '复制失败，请直接选择资料文字，使用系统复制功能。';
    } finally {
      button.disabled = false;
      if (restoreFocus && button.isConnected) button.focus({ preventScroll: true });
    }
  });
})();
