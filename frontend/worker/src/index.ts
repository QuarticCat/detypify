import { Hono } from "hono";
import { cors } from "hono/cors";

type Bindings = {
    DB: D1Database;
};

type ContribPayload = {
    ver: number;
    token: number;
    samples: [number, number][][];
};

const app = new Hono<{ Bindings: Bindings }>();

app.use("/*", cors());

app.post("/contrib", async (c) => {
    const { ver, token, samples } = await c.req.json<ContribPayload>();
    const stmt = c.env.DB.prepare("INSERT INTO samples (ver, token, sym, strokes) VALUES (?, ?, ?, ?)");
    const inserts = samples.map(([sym, strokes]) => stmt.bind(ver, token, sym, JSON.stringify(strokes)));
    await c.env.DB.batch(inserts);
    return c.text("Thanks for your contributions!");
});

export default app;
