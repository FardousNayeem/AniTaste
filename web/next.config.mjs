/** @type {import('next').NextConfig} */

// The Python API runs as its own process. Proxying it under /api means the
// browser only ever talks to one origin, so there is no CORS in production
// and no API URL baked into the client bundle.
const API = process.env.ANITASTE_API ?? "http://127.0.0.1:8770";

const nextConfig = {
  async rewrites() {
    return [{ source: "/api/:path*", destination: `${API}/api/:path*` }];
  },
};

export default nextConfig;
