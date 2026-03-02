import type { ReactNode } from "react";

import "./globals.css";

export const metadata = {
  title: "ComfyUI Inpaint Studio",
  description: "Run multiple ComfyUI inpainting pipelines on one image.",
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
