/**
 * Route Handler: proxies SSE stream from the backend without buffering.
 *
 * Why not use next.config.ts rewrites?
 * The rewrite proxy has a ~30s idle timeout. During reranking (30-60s of silence),
 * the proxy drops the connection and the frontend shows an error.
 *
 * This handler uses a TransformStream to pipe bytes from the backend directly
 * to the client with no buffering and no idle timeout.
 */

import { NextRequest } from "next/server";

const BACKEND = (
    process.env.API_REWRITE_TARGET ||
    process.env.NEXT_PUBLIC_API_URL ||
    "http://localhost:8000"
).replace(/\/+$/, "");

export const dynamic = "force-dynamic";

export async function POST(req: NextRequest) {
    const body = await req.text();
    const auth = req.headers.get("authorization") ?? "";
    const headers: Record<string, string> = { "Content-Type": "application/json" };
    if (auth.trim()) {
        headers.Authorization = auth;
    }

    let upstream: Response;
    try {
        upstream = await fetch(`${BACKEND}/api/v1/chat/stream`, {
            method: "POST",
            headers,
            body,
        });
    } catch (err) {
        return Response.json({ error: String(err) }, { status: 502 });
    }

    if (!upstream.body) {
        return Response.json({ error: "No body from backend" }, { status: 502 });
    }

    // Pipe the upstream ReadableStream directly — no buffering, no timeout
    const { readable, writable } = new TransformStream();
    upstream.body.pipeTo(writable).catch(() => {/* client disconnected */ });

    return new Response(readable, {
        status: upstream.status,
        headers: {
            "Content-Type": "text/event-stream; charset=utf-8",
            "Cache-Control": "no-cache, no-transform",
            Connection: "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    });
}
