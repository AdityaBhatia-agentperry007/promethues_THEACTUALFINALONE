import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Animated Explainer",
  description: "Physics concept explainer with authoring brief, Manim prompt, and rendered output"
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

