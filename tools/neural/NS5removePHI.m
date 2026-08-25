function NS5removePHI(nsxfilepath,savefilepath)
%% remove PHI from NS5 files in a task folder (Microphone data)
% 
% inputs: 
% nsxfilepath (full path of the task folder of interest, with ns5 and other data files)
% savefilepath (full path of the folder where the de-identified ns5 should
% be save)
% Example:
% nsxfilepath='S:\EMU-18112\YFQ\EMU-0016_subj-YFQ_task-PEP_lead-LT2aA_elec-1\';
% savefilepath='B:\Bartoli\DATA\';
% NS5removePHI(nsxfilepath,savefilepath)
% 
% note: savefilepath can be the same as nsxfilepath if nsxfilepath is not on stitched
% (you should not save new data in stitched folders)
% if you are reading the task folder directly from stitched, make sure that you save somewhere else
%
% overview: given a folder path (nsxfilepath) the code checks for any ".ns5" files, 
% loops through them, zeroing our any analog channel with the word "Mic" in
% the channel name, and saves the NS5 in a new folder path (savefilepath)
% with the same name as the original file and the suffix "noPHI"
% the NS5 is the same structure as the original one, minus the mic channels
%
% 1) open original NS5 (in the folder defined by nsxfilepath) 
% 2) remove data from analog channels containing the label Mic (zero out)
% 3) saveNS5 again in new location with suffix noPHI (in the savefilepath folder)
% 
% requires BRK Toolbox NPMK version '5.5.3.0' or less! 
%
% newer versions are not compatible with our TOC recordings and will cause 
% matlab to run out of space (they will attempt to zeropad data based on
% the assumptions that each recording starts at a timestap of 0, while our
% recodings have large timestamps reflecting the delay between the
% recording and the moment the NSPs where turned on. If you get the out of
% space warning, and your recording is not that long/you have decent RAM,
% then it's likely you are using the wrong version of NPMK
%
% Eleonora Aug 2025

%% inputs to change

% folder w original file(s):
% nsxfilepath='B:\Bartoli\People\LM\helping_friends\EMU-0092_subj-YFM_task-anticipation_run-01\';

% folder where you want to save the modified file(s):
% savefilepath='B:\Bartoli\DATA\';

%% body of code

cd(nsxfilepath)

nsxfilesindir=dir(nsxfilepath);
ns5indx=contains({nsxfilesindir.name},'.ns5');
ns5pos=find(ns5indx);

if ~isempty(ns5pos)
    for nfiles=1:numel(ns5pos)
        fprintf('Loading %s:\n',nsxfilesindir(ns5pos(nfiles)).name)
        openNSx([nsxfilepath nsxfilesindir(ns5pos(nfiles)).name])
        mic = contains({NS5.ElectrodesInfo.Label},'Mic');
        if ~iscell(NS5.Data)
            NS5.Data(mic,:)=zeros(size(NS5.Data(mic,:)));
        else
            for ncells=1:numel(NS5.Data)
                NS5.Data{ncells}(mic,:)=zeros(size(NS5.Data{ncells}(mic,:)));
            end
        end
        cd(savefilepath)
        % saveNSx: second argument is optional and is the filename
        fprintf('Saving %s:\n',['noPHI' nsxfilesindir(ns5pos(nfiles)).name])
        saveNSx(NS5,['noPHI' nsxfilesindir(ns5pos(nfiles)).name])
        clear NS5
        cd(nsxfilepath)
    end
else
    fprintf('No .ns5 files found at %s: check address again\n',nsxfilepath)
end

% check:
% cd(nsxfilepath)
% openNSx([nsxfilepath nsxfilesindir(ns5pos(nfiles)).name])
% original=NS5.Data;
% clear NS5
% cd(savefilepath)
% openNSx(['noPHI' nsxfilesindir(ns5pos(nfiles)).name])
% modified = NS5.Data;
% compare original and new: nothing should have changed except on mic
% channels
% figure; plot(original(1,:),modified(1,:)); 