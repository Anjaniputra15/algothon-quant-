/** @type {import('next').NextConfig} */
const nextConfig = {
  // Disable static export to prevent SSR issues
  output: 'standalone',
  
  // Configure webpack to handle Plotly properly
  webpack: (config, { isServer }) => {
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
        net: false,
        tls: false,
      };
    }
    return config;
  },
};

module.exports = nextConfig; 