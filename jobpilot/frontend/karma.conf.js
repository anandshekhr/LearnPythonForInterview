// Headless Chrome for CI/containers. CHROME_BIN can point at any Chromium build.
module.exports = function (config) {
  config.set({
    basePath: '',
    frameworks: ['jasmine', '@angular-devkit/build-angular'],
    plugins: [require('karma-jasmine'), require('karma-chrome-launcher'), require('@angular-devkit/build-angular/plugins/karma')],
    browsers: ['ChromeHeadlessCI'],
    customLaunchers: { ChromeHeadlessCI: { base: 'ChromeHeadless', flags: ['--no-sandbox', '--disable-dev-shm-usage'] } },
    singleRun: true,
    restartOnFileChange: false,
  });
};
