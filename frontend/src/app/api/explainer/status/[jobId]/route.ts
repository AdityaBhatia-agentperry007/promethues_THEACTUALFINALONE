const BACKEND = process.env.BACKEND_URL ?? "http://127.0.0.1:8000";

export const dynamic = "force-dynamic";

const METADATA_MAP: Record<string, { title: string, visual_goal: string, video: string }> = {
  "post_neutron_star_merger": {
    title: "Black Hole Accretion Disk",
    visual_goal: "create a black hole accretion disk in deep space",
    video: "blackhole_intro.mp4"
  },
  "supernova_explosion_64": {
    title: "Supernova Shock Front",
    visual_goal: "predict a supernova blast wave through turbulent interstellar gas",
    video: "supernova_intro.mp4"
  },
  "rayleigh_benard": {
    title: "Thermal Convection Plumes",
    visual_goal: "show Rayleigh-Benard thermal convection plumes",
    video: "convection_intro.mp4"
  },
  "acoustic_scattering_maze": {
    title: "Acoustic Maze Scattering",
    visual_goal: "simulate acoustic wave scattering through a maze",
    video: "acoustic_intro.mp4"
  },
  "planetswe": {
    title: "Planetary Vortex Dynamics",
    visual_goal: "predict shallow-water storm vortex and jet stream",
    video: "planet_intro.mp4"
  },
  "MHD_64": {
    title: "MHD Wave Turbulence",
    visual_goal: "supersonic sub-Alfvenic wave turbulence in plasma",
    video: "mhd_intro.mp4"
  }
};

export async function GET(request: Request, context: { params: Promise<{ jobId: string }> }) {
  const { jobId } = await context.params;

  const host = request.headers.get("host") ?? "localhost:3000";
  const protocol = host.startsWith("localhost") || host.startsWith("127.0.0.1") ? "http" : "https";
  const origin = `${protocol}://${host}`;

  const isMock = jobId.startsWith("mock_job_");

  if (!isMock) {
    try {
      const upstream = await fetch(`${BACKEND}/explainer/status/${encodeURIComponent(jobId)}`, {
        method: "GET",
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
      console.warn("Backend explainer offline, serving mock fallback state for job:", jobId, error);
      // Fall through to mock handling
    }
  }

  // Handle mock fallback response
  const dataset = isMock ? jobId.replace("mock_job_", "") : "MHD_64";
  const meta = METADATA_MAP[dataset] ?? METADATA_MAP["MHD_64"];

  return Response.json({
    job_id: jobId,
    status: "done_fallback",
    url: `${origin}/fallback/${meta.video}`,
    route: {
      recommended_dataset: dataset,
      reason: "The prompt was mapped to the closest The Well physics domain."
    },
    label: "CONCEPT ANIMATION | generated illustration | anchored to The Well",
    attempts: 0,
    provider: "fallback_mock",
    explanation: `Animated Explainer: Mock simulation fallback. This is a generated concept illustration, not predicted simulation field data.`,
    brief: {
      title: meta.title,
      visual_goal: meta.visual_goal,
      anchor_dataset: dataset
    }
  });
}
