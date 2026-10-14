function appendRunLog(runDir, text)
%APPENDRUNLOG append text to <runDir>/run.log (UTF-8 plain text).
%   each call opens-writes-closes, content is flushed immediately, convenient for tail -f during a run.
%   text can be char or string and may contain newlines.
    if isempty(runDir)
        return;
    end
    if isstring(text)
        text = char(text);
    end
    fid = fopen(fullfile(runDir, 'run.log'), 'a', 'n', 'UTF-8');
    if fid < 0
        return;
    end
    fprintf(fid, '%s\n', text);
    fclose(fid);
end