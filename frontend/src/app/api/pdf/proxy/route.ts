import { NextRequest } from "next/server";

export const dynamic = "force-dynamic";

export async function GET(req: NextRequest) {
  const rawUrl = req.nextUrl.searchParams.get("url")?.trim();
  if (!rawUrl) {
    return Response.json({ error: "Missing url query param" }, { status: 400 });
  }

  let parsed: URL;
  try {
    parsed = new URL(rawUrl);
  } catch {
    return Response.json({ error: "Invalid url" }, { status: 400 });
  }

  if (!["http:", "https:"].includes(parsed.protocol)) {
    return Response.json({ error: "Only http/https are supported" }, { status: 400 });
  }

  let upstream: Response;
  try {
    upstream = await fetch(parsed.toString(), {
      redirect: "follow",
      cache: "no-store",
    });
  } catch (error) {
    return Response.json({ error: `Failed to fetch upstream URL: ${String(error)}` }, { status: 502 });
  }

  if (!upstream.body) {
    return Response.json({ error: "No response body from upstream URL" }, { status: 502 });
  }

  const headers = new Headers(upstream.headers);
  headers.delete("content-security-policy");
  headers.delete("x-frame-options");
  headers.delete("frame-options");
  headers.delete("set-cookie");
  headers.set("Cache-Control", "no-store");

  return new Response(upstream.body, {
    status: upstream.status,
    headers,
  });
}
