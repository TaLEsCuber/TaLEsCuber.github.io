/**
 * encrypted-files.js — Client-side decryption for .enc files
 *
 * Handles two encryption formats:
 *   Type 0 (PBKDF2):  [type:1] [iterations:4 LE] [salt:32] [IV:16] [ciphertext]
 *   Type 1 (site key): [type:1] [IV:16] [ciphertext]
 *
 * Both prepend "ENCF" magic to plaintext before encryption.
 */

(function() {
  'use strict';

  var ENCF_MAGIC = [69, 78, 67, 70]; // "ENCF"

  // ─── MIME type lookup ──────────────────────────────────

  function mimeType(filename) {
    var ext = filename.split('.').pop().toLowerCase();
    var map = {
      pdf: 'application/pdf',
      png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg',
      gif: 'image/gif', webp: 'image/webp', svg: 'image/svg+xml',
      doc: 'application/msword', docx: 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
      xls: 'application/vnd.ms-excel', xlsx: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
      zip: 'application/zip',
      txt: 'text/plain', json: 'application/json',
      mp4: 'video/mp4', mp3: 'audio/mpeg'
    };
    return map[ext] || 'application/octet-stream';
  }

  // ─── Core decryption ────────────────────────────────────

  /**
   * Fetch and decrypt a .enc file.
   * @param {string}  encUrl   — URL of the .enc file
   * @param {string}  password — Plaintext password (for type 0 / PBKDF2)
   * @param {boolean} isPbkdf2 — true = PBKDF2 mode, false/omitted = site-key mode
   * @returns {Promise<{blobUrl: string, filename: string}>}
   */
  async function fetchAndDecrypt(encUrl, password, isPbkdf2) {
    var resp = await fetch(encUrl);
    if (!resp.ok) throw new Error('Failed to fetch: ' + encUrl);
    var buf = await resp.arrayBuffer();

    var type = new Uint8Array(buf)[0];
    var ciphertext, iv;

    if (isPbkdf2 || type === 0) {
      // PBKDF2 format: [type:1] [iterations:4 LE] [salt:32] [IV:16] [ciphertext]
      var headerView = new DataView(buf);
      var iterations = headerView.getUint32(1, true);
      var salt = new Uint8Array(buf, 5, 32);
      iv = new Uint8Array(buf, 37, 16);
      ciphertext = new Uint8Array(buf, 53);

      var enc = new TextEncoder();
      var passwordKey = await crypto.subtle.importKey('raw', enc.encode(password), 'PBKDF2', false, ['deriveKey']);
      var aesKey = await crypto.subtle.deriveKey(
        { name: 'PBKDF2', salt: salt, iterations: iterations, hash: 'SHA-256' },
        passwordKey,
        { name: 'AES-CBC', length: 256 },
        false,
        ['decrypt']
      );
    } else {
      // Site-key format: [type:1] [IV:16] [ciphertext]
      iv = new Uint8Array(buf, 1, 16);
      ciphertext = new Uint8Array(buf, 17);

      // Import raw site key
      var keyBytes = hexToBytes(window.__SITE_ENC_KEY__);
      var aesKey = await crypto.subtle.importKey('raw', keyBytes, { name: 'AES-CBC' }, false, ['decrypt']);
    }

    var decrypted = await crypto.subtle.decrypt({ name: 'AES-CBC', iv: iv }, aesKey, ciphertext);
    var decBytes = new Uint8Array(decrypted);

    // Verify magic
    if (decBytes[0] !== ENCF_MAGIC[0] || decBytes[1] !== ENCF_MAGIC[1] ||
        decBytes[2] !== ENCF_MAGIC[2] || decBytes[3] !== ENCF_MAGIC[3]) {
      throw new Error('BAD_MAGIC');
    }

    var plaintext = decBytes.slice(4);
    var origFilename = encUrl.split('/').pop().replace(/\.enc$/, '');
    var blob = new Blob([plaintext], { type: mimeType(origFilename) });
    var blobUrl = URL.createObjectURL(blob);

    return { blobUrl: blobUrl, filename: origFilename };
  }

  function hexToBytes(hex) {
    var bytes = new Uint8Array(hex.length / 2);
    for (var i = 0; i < hex.length; i += 2) {
      bytes[i / 2] = parseInt(hex.substr(i, 2), 16);
    }
    return bytes;
  }

  // ─── Auto-decrypt public PDF links/embeds ───────────────

  function setupAutoDecrypt() {
    if (!window.__SITE_ENC_KEY__) return;

    // Intercept clicks on links to .enc files
    document.addEventListener('click', function(e) {
      var link = e.target.closest('a');
      if (!link) return;
      var href = link.getAttribute('href') || '';
      if (!href.endsWith('.enc')) return;

      e.preventDefault();
      var el = link;

      // Show loading
      var origText = el.textContent;
      el.textContent = '解密中...';
      el.style.opacity = '0.6';
      el.style.pointerEvents = 'none';

      fetchAndDecrypt(href, null, false).then(function(result) {
        // Trigger download
        var a = document.createElement('a');
        a.href = result.blobUrl;
        a.download = result.filename;
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);

        // Restore button
        el.textContent = origText;
        el.style.opacity = '';
        el.style.pointerEvents = '';
      }).catch(function(err) {
        console.error('Decrypt failed:', err);
        el.textContent = origText;
        el.style.opacity = '';
        el.style.pointerEvents = '';
        alert('文件解密失败，请刷新页面后重试。');
      });
    });

    // Replace .enc PDF embeds with auto-decrypt placeholders
    document.querySelectorAll('embed[src$=".enc"], iframe[src$=".enc"]').forEach(function(el) {
      var encUrl = el.getAttribute('src');
      var container = document.createElement('div');
      container.className = 'enc-pdf-placeholder';
      container.innerHTML =
        '<div style="text-align:center;padding:40px;background:#f5f5f5;border-radius:8px;color:#666;">' +
          '<p>加密 PDF — 正在解密...</p>' +
        '</div>';
      el.parentNode.replaceChild(container, el);

      fetchAndDecrypt(encUrl, null, false).then(function(result) {
        var embed = document.createElement('embed');
        embed.src = result.blobUrl;
        embed.type = 'application/pdf';
        embed.width = '100%';
        embed.height = '600px';
        embed.style.border = 'none';
        container.parentNode.replaceChild(embed, container);
      }).catch(function(err) {
        console.error('PDF decrypt failed:', err);
        container.innerHTML =
          '<div style="text-align:center;padding:40px;background:#fff0f0;border-radius:8px;color:#c33;">' +
            '<p>PDF 解密失败，请刷新页面后重试。</p>' +
          '</div>';
      });
    });
  }

  // ─── Expose for countdown calendar ──────────────────────

  window.__decryptFilePBKDF2 = function(encUrl, password) {
    return fetchAndDecrypt(encUrl, password, true);
  };

  window.__decryptFileSiteKey = function(encUrl) {
    return fetchAndDecrypt(encUrl, null, false);
  };

  // ─── Init ───────────────────────────────────────────────

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupAutoDecrypt);
  } else {
    setupAutoDecrypt();
  }

})();
