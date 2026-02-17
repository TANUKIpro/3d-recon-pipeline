import { defineConfig } from 'vitest/config';
import { resolve } from 'path';

const threeStub = resolve(__dirname, 'tests/dashboard/frontend/helpers/three-stub.js');

export default defineConfig({
  resolve: {
    alias: {
      'three/addons/controls/OrbitControls.js': threeStub,
      'three/addons/loaders/PLYLoader.js': threeStub,
      'three/addons/loaders/OBJLoader.js': threeStub,
      'three/addons/loaders/MTLLoader.js': threeStub,
      'three': threeStub,
    },
  },
  test: {
    environment: 'jsdom',
    include: ['tests/dashboard/frontend/**/*.test.js'],
    setupFiles: ['tests/dashboard/frontend/helpers/setup.js'],
  },
});
