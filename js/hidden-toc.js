(function () {
  'use strict';

  // This script can run again after PJAX navigation; install global hooks once.
  if (window.hiddenArticleToc) {
    window.hiddenArticleToc();
    return;
  }

  var currentCard;
  var headings = [];
  var frame;

  function updateActive() {
    frame = null;
    if (!currentCard || !currentCard.isConnected) return;
    var active = headings[0];
    headings.forEach(function (entry) {
      if (entry.heading.getBoundingClientRect().top <= 80) active = entry;
    });
    currentCard.querySelectorAll('.active').forEach(function (element) {
      element.classList.remove('active');
    });
    if (!active) return;
    active.link.classList.add('active');
    for (var parent = active.link.parentElement; parent && !parent.classList.contains('toc'); parent = parent.parentElement) {
      if (parent.tagName === 'LI') parent.classList.add('active');
    }
  }

  function install() {
    var template = document.querySelector('#hexo-blog-encrypt template#hidden-article-toc');
    var card = document.querySelector('#card-toc .toc-content');
    if (!template || !card) return;
    card.replaceChildren(template.content.cloneNode(true));
    card.style.display = 'block';
    template.remove();
    currentCard = card;
    headings = Array.from(card.querySelectorAll('.toc-link[href^="#"]')).map(function (link) {
      return { link: link, heading: document.getElementById(decodeURIComponent(link.getAttribute('href').slice(1))) };
    }).filter(function (entry) { return entry.heading; });
    updateActive();
  }

  window.hiddenArticleToc = install;
  window.addEventListener('hexo-blog-decrypt', install);
  document.addEventListener('pjax:complete', install);
  window.addEventListener('scroll', function () {
    if (!frame) frame = window.requestAnimationFrame(updateActive);
  }, { passive: true });
  install();
})();
