if(!self.define){let e,i={};const n=(n,s)=>(n=new URL(n+".js",s).href,i[n]||new Promise((i=>{if("document"in self){const e=document.createElement("script");e.src=n,e.onload=i,document.head.appendChild(e)}else e=n,importScripts(n),i()})).then((()=>{let e=i[n];if(!e)throw new Error(`Module ${n} didn’t register its module`);return e})));self.define=(s,r)=>{const c=e||("document"in self?document.currentScript.src:"")||location.href;if(i[c])return;let d={};const o=e=>n(e,c),a={module:{uri:c},exports:d,require:o};i[c]=Promise.all(s.map((e=>a[e]||o(e)))).then((e=>(r(...e),d)))}}define(["./workbox-2dfdff9b"],(function(e){"use strict";self.skipWaiting(),e.clientsClaim(),e.precacheAndRoute([{url:"apple-touch-icon.png",revision:"736516a7a4b7b9e554e2fc76bd393afe"},{url:"assets/index-BV4ujCFg.css",revision:null},{url:"assets/index-Dm6YRw4T.js",revision:null},{url:"assets/model-C_C_cCv5.onnx",revision:null},{url:"favicon.ico",revision:"06e98a287580adefdb12d65e5e62b992"},{url:"favicon.svg",revision:"1f738e984048350ac3a4b097508444e4"},{url:"index.html",revision:"96cd81478957a057e4e39cd49e14cf16"},{url:"NewCMMath-Detypify.woff2",revision:"2f92af13878423348440805d6a6b01c2"},{url:"pwa-192x192.png",revision:"6da11aa76085d626adc705faae515032"},{url:"pwa-512x512.png",revision:"5a6d2a2327849170366e7c4c8c9d22d1"},{url:"registerSW.js",revision:"1872c500de691dce40960bb85481de07"},{url:"pwa-192x192.png",revision:"6da11aa76085d626adc705faae515032"},{url:"pwa-512x512.png",revision:"5a6d2a2327849170366e7c4c8c9d22d1"},{url:"manifest.webmanifest",revision:"6ecc8d3cc3bb9767880db883aa6c8965"}],{}),e.cleanupOutdatedCaches(),e.registerRoute(new e.NavigationRoute(e.createHandlerBoundToURL("index.html"))),e.registerRoute((({url:e})=>"https://cdn.jsdelivr.net"===e.origin),new e.CacheFirst({cacheName:"jsdelivr",plugins:[]}),"GET")}));