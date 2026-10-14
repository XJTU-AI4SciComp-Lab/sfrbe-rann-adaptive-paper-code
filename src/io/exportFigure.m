function exportFigure(fig, outPdf, outPng)
%EXPORTFIGURE unified export: compressed PDF (150dpi embedded image) + PNG (300dpi), then close the figure.
%   canvas is 720x580 points, consistent with the paper plotting style.
    set(fig, 'PaperUnits', 'points', ...
        'PaperSize', [720, 580], ...
        'PaperPosition', [0, 0, 720, 580]);
    exportgraphics(fig, outPdf, 'ContentType', 'image', 'Resolution', 150);
    try
        print(fig, outPng, '-dpng', '-r300');
    catch
        exportgraphics(fig, outPng, 'Resolution', 300);
    end
    close(fig);
end