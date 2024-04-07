import { Hono } from "hono";
import { cors } from "hono/cors";

const app = new Hono();

app.use("/*", cors());

app.post("/contrib", async (c) => {
    let { ver, token, samples } = await c.req.json();
    let stmt = c.env.DB.prepare("INSERT INTO samples (ver, token, sym, strokes) VALUES (?, ?, ?, ?)");
    let inserts = samples.map(([sym, strokes]) => stmt.bind(ver, token, sym, JSON.stringify(strokes)));
    await c.env.DB.batch(inserts);
    return c.text("Success");
});

export default app;
