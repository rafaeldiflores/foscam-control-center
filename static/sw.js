// Foscam PWA Service Worker
const CACHE_NAME = 'foscam-pwa-v1';
const STATIC_ASSETS = [
  '/',
  '/manifest.json',
  '/static/icon-192.png',
  '/static/icon-512.png'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return cache.addAll(STATIC_ASSETS);
    })
  );
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => {
      return Promise.all(
        keys.filter((key) => key !== CACHE_NAME).map((key) => caches.delete(key))
      );
    })
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Bypass cache completely for API, WebSockets, and Video streams
  if (url.pathname.startsWith('/api/') || 
      url.pathname.startsWith('/ws/') || 
      url.pathname.startsWith('/recordings/') || 
      url.pathname.startsWith('/snapshots/')) {
    return event.respondWith(fetch(event.request));
  }

  // Network first with cache fallback for static app shell
  event.respondWith(
    fetch(event.request)
      .then((response) => {
        if (response.status === 200) {
          const resClone = response.clone();
          caches.open(CACHE_NAME).then((cache) => {
            cache.put(event.request, resClone);
          });
        }
        return response;
      })
      .catch(() => caches.match(event.request))
  );
});
