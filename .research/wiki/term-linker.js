(function () {
  "use strict";

  const registry = window.RESEARCH_WIKI;
  if (!registry || !Array.isArray(registry.terms)) return;

  const scriptUrl = new URL(document.currentScript.src, location.href);
  const wikiUrl = new URL("index.html", scriptUrl);
  const aliasToId = new Map();
  registry.terms.forEach(term => term.aliases.forEach(alias => {
    const key = alias.toLocaleLowerCase();
    if (!aliasToId.has(key)) aliasToId.set(key, term.id);
  }));
  const aliases = [...aliasToId.keys()].sort((a, b) => b.length - a.length);
  const escapeRegExp = value => value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const matcher = new RegExp(`(^|[^\\p{L}\\p{N}])(${aliases.map(escapeRegExp).join("|")})(?=$|[^\\p{L}\\p{N}])`, "giu");
  const skipped = new Set(["A", "BUTTON", "CODE", "PRE", "SCRIPT", "STYLE", "TEXTAREA", "SVG", "MATH", "OPTION"]);

  const style = document.createElement("style");
  style.textContent = ".wiki-term{border-bottom:1px dotted currentColor;color:inherit;text-decoration:none}.wiki-term:hover{color:var(--blue,#0b6fc2);border-bottom-style:solid}";
  document.head.append(style);

  function termHref(id) {
    const target = new URL(wikiUrl.href);
    target.searchParams.set("return", location.pathname + location.search + location.hash);
    target.hash = id;
    return target.href;
  }

  function linkTextNode(node) {
    if (!node.nodeValue || !node.nodeValue.trim()) return;
    const parent = node.parentElement;
    if (!parent || skipped.has(parent.tagName) || parent.closest("a, button, code, pre, script, style, textarea, svg, math, [data-no-wiki]")) return;
    matcher.lastIndex = 0;
    const text = node.nodeValue;
    let match, last = 0, changed = false;
    const fragment = document.createDocumentFragment();
    while ((match = matcher.exec(text))) {
      const prefix = match[1];
      const label = match[2];
      const start = match.index + prefix.length;
      const end = start + label.length;
      const id = aliasToId.get(label.toLocaleLowerCase());
      if (!id) continue;
      fragment.append(document.createTextNode(text.slice(last, start)));
      const link = document.createElement("a");
      link.className = "wiki-term";
      link.href = termHref(id);
      link.textContent = label;
      link.title = `Open the research wiki definition for ${label}`;
      link.setAttribute("aria-label", `${label}: open definition in the research wiki`);
      link.addEventListener("click", () => {
        try {
          sessionStorage.setItem("research-wiki-return", JSON.stringify({url:location.href, title:document.title, scrollY:window.scrollY}));
        } catch (_) {}
      });
      fragment.append(link);
      last = end;
      changed = true;
    }
    if (!changed) return;
    fragment.append(document.createTextNode(text.slice(last)));
    node.replaceWith(fragment);
  }

  function linkTree(root) {
    if (root.nodeType === Node.TEXT_NODE) {
      linkTextNode(root);
      return;
    }
    if (root.nodeType !== Node.ELEMENT_NODE || skipped.has(root.tagName) || root.matches("[data-no-wiki]")) return;
    const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT);
    const nodes = [];
    while (walker.nextNode()) nodes.push(walker.currentNode);
    nodes.forEach(linkTextNode);
  }

  function restoreScrollIfNeeded() {
    try {
      const raw = sessionStorage.getItem("research-wiki-restore");
      if (!raw) return;
      const state = JSON.parse(raw);
      if (state.url !== location.href) return;
      sessionStorage.removeItem("research-wiki-restore");
      requestAnimationFrame(() => requestAnimationFrame(() => window.scrollTo({top:state.scrollY || 0, behavior:"auto"})));
    } catch (_) {}
  }

  linkTree(document.body);
  restoreScrollIfNeeded();
  const observer = new MutationObserver(records => records.forEach(record => record.addedNodes.forEach(linkTree)));
  observer.observe(document.body, {childList:true, subtree:true});
})();
