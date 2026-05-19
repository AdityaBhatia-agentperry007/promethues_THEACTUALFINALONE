const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export const dynamic = "force-dynamic";

export async function POST(request: Request) {
  let body = "";
  try {
    body = await request.text();
  } catch (err) {
    // Empty body
  }

  // Determine target fallback dataset from prompt
  let dataset = "MHD_64";
  try {
    const json = JSON.parse(body);
    const prompt = (json.prompt || "").toLowerCase();
    if (prompt.includes("black hole") || prompt.includes("blackhole") || prompt.includes("neutron") || prompt.includes("merger") || prompt.includes("accretion") || prompt.includes("lensing")) {
      dataset = "post_neutron_star_merger";
    } else if (prompt.includes("supernova") || prompt.includes("blast") || prompt.includes("shock") || prompt.includes("explosion")) {
      dataset = "supernova_explosion_64";
    } else if (prompt.includes("convection") || prompt.includes("rayleigh") || prompt.includes("benard") || prompt.includes("plume")) {
      dataset = "rayleigh_benard";
    } else if (prompt.includes("acoustic") || prompt.includes("wave") || prompt.includes("maze") || prompt.includes("scattering") || prompt.includes("sound")) {
      dataset = "acoustic_scattering_maze";
    } else if (prompt.includes("planet") || prompt.includes("shallow") || prompt.includes("vortex") || prompt.includes("storm")) {
      dataset = "planetswe";
    }
  } catch (e) {
    // Ignore JSON parsing errors
  }

  try {
    const upstream = await fetch(`${BACKEND}/explainer/animate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body,
      cache: "no-store"
    });
    
    if (!upstream.ok) {
      throw new Error(`Upstream status ${upstream.status}`);
    }

    const text = await upstream.text();
    return new Response(text, {
      status: upstream.status,
      headers: { "Content-Type": upstream.headers.get("Content-Type") ?? "application/json" }
    });
  } catch (error) {
    console.warn("Backend explainer offline, serving mock fallback flow:", error);
    // Return mock successful queued response
    return Response.json({
      job_id: `mock_job_${dataset}`,
      status: "queued",
      provider: "fallback_mock"
    });
  }
}
