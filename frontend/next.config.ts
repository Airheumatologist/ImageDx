import type { NextConfig } from "next";

const apiRewriteTarget = (
  process.env.API_REWRITE_TARGET ??
  process.env.NEXT_PUBLIC_API_URL ??
  "http://localhost:8000"
).replace(/\/+$/, "");

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiRewriteTarget}/api/:path*`,
      },
    ];
  },
};

export default nextConfig;
