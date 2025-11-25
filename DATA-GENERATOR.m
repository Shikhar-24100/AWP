%% Antenna Dataset Generator - DIPOLE ONLY
% Generates 2000 dipole antenna samples
% Uses MATLAB Antenna Toolbox

clear all; close all; clc;

%% Check for Antenna Toolbox
if ~license('test', 'Antenna_Toolbox')
    error('Antenna Toolbox not available. Please install it.');
end

%% Configuration
num_samples = 3400;
angles = 0:1:359;
c = physconst('LightSpeed');

%% Initialize
all_data = [];

fprintf('======================================================\n');
fprintf('DIPOLE ANTENNA DATASET GENERATION\n');
fprintf('======================================================\n');
fprintf('Total samples: %d\n', num_samples);
fprintf('Starting generation...\n\n');

tic;

%% ========== DIPOLE ANTENNAS ==========
fprintf('Generating DIPOLE antennas...\n');
for i = 1:num_samples
    try
        % Random frequency
        freq = rand() * (3e9 - 300e6) + 300e6;
        wavelength = c / freq;
        
        % Random length (0.25λ to 2λ)
        length_wl = rand() * (2.0 - 0.25) + 0.25;
        length = length_wl * wavelength;
        
        % Random width (0.0001λ to 0.01λ)
        width_wl = rand() * (0.01 - 0.0001) + 0.0001;
        width = width_wl * wavelength;
        
        % Create dipole
        ant = dipole('Length', length, 'Width', width);
        
        % Calculate radiation pattern
        gain_pattern = pattern(ant, freq, 0, angles, 'Type', 'directivity');
        
        % Calculate metrics
        max_gain = max(gain_pattern);
        min_gain = min(gain_pattern);
        avg_gain = mean(gain_pattern);
        
        % Beamwidth (HPBW)
        half_power = max_gain - 3;
        above_half = gain_pattern >= half_power;
        beamwidth = sum(above_half);
        
        % Store data
        row_data = [i, freq/1e9, wavelength, ...
                    length, length_wl, width, width_wl, ...
                    max_gain, min_gain, avg_gain, beamwidth, ...
                    gain_pattern'];
        
        all_data = [all_data; row_data];
        
        % Progress
        if mod(i, 100) == 0
            fprintf('  Progress: %d/%d (%.1f%% complete)\n', i, num_samples, (i/num_samples)*100);
        end
        
    catch ME
        fprintf('  Warning: Sample %d failed - %s\n', i, ME.message);
        continue;
    end
end

elapsed_time = toc;

%% ========== SAVE DATASET ==========
fprintf('\n======================================================\n');
fprintf('SAVING DATASET...\n');
fprintf('======================================================\n');

% Create headers
headers = {'sample_id', 'frequency_ghz', 'wavelength_m', ...
           'length_m', 'length_wl', 'width_m', 'width_wl', ...
           'max_gain_dbi', 'min_gain_dbi', 'avg_gain_dbi', 'beamwidth_deg'};

for angle = 0:359
    headers{end+1} = sprintf('gain_%ddeg', angle);
end

% Convert to table
T = array2table(all_data, 'VariableNames', headers);

% Save
output_file = 'dipole_dataset.csv';
writetable(T, output_file);

fprintf('✅ Dataset saved: %s\n', output_file);
fprintf('   Total samples: %d\n', height(T));
fprintf('   Total columns: %d\n', width(T));
fprintf('   Generation time: %.1f minutes\n\n', elapsed_time/60);

%% ========== STATISTICS ==========
fprintf('======================================================\n');
fprintf('DATASET STATISTICS\n');
fprintf('======================================================\n');

fprintf('\nFrequency range:\n');
fprintf('  Min: %.3f GHz\n', min(T.frequency_ghz));
fprintf('  Max: %.3f GHz\n', max(T.frequency_ghz));
fprintf('  Mean: %.3f GHz\n', mean(T.frequency_ghz));

fprintf('\nLength (wavelengths):\n');
fprintf('  Min: %.3f λ\n', min(T.length_wl));
fprintf('  Max: %.3f λ\n', max(T.length_wl));
fprintf('  Mean: %.3f λ\n', mean(T.length_wl));

fprintf('\nGain statistics:\n');
fprintf('  Max gain range: %.2f to %.2f dBi\n', min(T.max_gain_dbi), max(T.max_gain_dbi));
fprintf('  Average max gain: %.2f dBi\n', mean(T.max_gain_dbi));
fprintf('  Average beamwidth: %.1f degrees\n', mean(T.beamwidth_deg));

%% ========== VISUALIZATION ==========
fprintf('\n======================================================\n');
fprintf('CREATING VISUALIZATIONS...\n');
fprintf('======================================================\n');

% Plot 1: Sample radiation patterns
figure('Position', [100 100 1400 600]);

for i = 1:6
    sample_idx = randi(height(T));
    
    subplot(2, 3, i);
    gain_data = table2array(T(sample_idx, 12:end));
    
    polarplot(deg2rad(angles), gain_data, 'LineWidth', 2);
    title(sprintf('Sample %d: %.2f GHz, %.2f λ\nMax Gain: %.2f dBi', ...
                  sample_idx, T.frequency_ghz(sample_idx), ...
                  T.length_wl(sample_idx), T.max_gain_dbi(sample_idx)));
    ax = gca;
    ax.ThetaZeroLocation = 'top';
    ax.ThetaDir = 'clockwise';
end

sgtitle('Sample Dipole Radiation Patterns', 'FontSize', 14, 'FontWeight', 'bold');
saveas(gcf, 'dipole_sample_patterns.png');
fprintf('✅ Sample patterns saved\n');

% Plot 2: Parameter analysis
figure('Position', [100 100 1400 900]);

subplot(2, 3, 1);
histogram(T.frequency_ghz, 40);
xlabel('Frequency (GHz)');
ylabel('Count');
title('Frequency Distribution');
grid on;

subplot(2, 3, 2);
histogram(T.length_wl, 40);
xlabel('Length (wavelengths)');
ylabel('Count');
title('Length Distribution');
grid on;

subplot(2, 3, 3);
histogram(T.max_gain_dbi, 40);
xlabel('Max Gain (dBi)');
ylabel('Count');
title('Gain Distribution');
grid on;

subplot(2, 3, 4);
scatter(T.length_wl, T.max_gain_dbi, 20, T.frequency_ghz, 'filled');
xlabel('Length (wavelengths)');
ylabel('Max Gain (dBi)');
title('Gain vs Length');
colorbar;
colormap('jet');
grid on;

subplot(2, 3, 5);
scatter(T.frequency_ghz, T.max_gain_dbi, 20, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Frequency (GHz)');
ylabel('Max Gain (dBi)');
title('Gain vs Frequency');
grid on;

subplot(2, 3, 6);
scatter(T.length_wl, T.beamwidth_deg, 20, 'filled', 'MarkerFaceAlpha', 0.5);
xlabel('Length (wavelengths)');
ylabel('Beamwidth (degrees)');
title('Beamwidth vs Length');
grid on;

saveas(gcf, 'dipole_parameter_analysis.png');
fprintf('✅ Parameter analysis saved\n');

fprintf('\n======================================================\n');
fprintf('✨ DATASET GENERATION COMPLETE!\n');
fprintf('======================================================\n');
fprintf('Generated:\n');
fprintf('  ✓ 2000 dipole samples\n');
fprintf('  ✓ Full 360° radiation patterns\n');
fprintf('  ✓ Frequency range: 300 MHz - 3 GHz\n');
fprintf('  ✓ Length range: 0.25λ - 2λ\n\n');
fprintf('Output files:\n');
fprintf('  • dipole_dataset.csv\n');
fprintf('  • dipole_sample_patterns.png\n');
fprintf('  • dipole_parameter_analysis.png\n');
fprintf('======================================================\n');
