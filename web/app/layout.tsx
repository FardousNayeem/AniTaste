import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "AniTaste - what should I watch?",
  description:
    "Anime recommendations from a model trained on the community recommendation graph.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
