import type { ReactNode } from "react";

import "./globals.css";

export const metadata = {
  title: "SemProbe – Semantic Robustness Probing via Inpainting",
  description: "An interactive tool for data augmentation for safety-critical object detection.",
};

type RootLayoutProps = {
  children: ReactNode;
};

export default function RootLayout({ children }: RootLayoutProps) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
