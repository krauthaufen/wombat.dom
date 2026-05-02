// Vite asset-URL imports.
declare module "*.ttf?url" { const url: string; export default url; }
declare module "*.otf?url" { const url: string; export default url; }
