'''
3D Slicer module: MR -> synthetic CT with the SAdaNGF generator.

All the model code and pre/post-processing live in MRtoCTLib/mr_to_sct.py (the same standalone file that runs
from the command line). The trained weights are not shipped with the extension: they are downloaded
once, on first use, from WEIGHTS_URL into Slicer's cache folder.

Creative Commons Attribution-NonCommercial 4.0 International
Copyright (c) 2026 Kevin Estiven Giraldo Paniagua
'''
import hashlib
import logging
import os
import time
import urllib.request

import ctk
import numpy as np
import qt
import slicer
import vtk
from slicer.ScriptedLoadableModule import (
    ScriptedLoadableModule,
    ScriptedLoadableModuleLogic,
    ScriptedLoadableModuleTest,
    ScriptedLoadableModuleWidget,
)

# ---------------------------------------------------------------------------------------------
# Model weights: fill in after uploading best_net_G_A.pth as a GitHub release asset (see README)
# ---------------------------------------------------------------------------------------------
WEIGHTS_FILENAME = 'best_net_G_A.pth'
WEIGHTS_URL = 'https://github.com/kgiraldo09/SlicerMRtoCT/releases/download/v1.0/best_net_G_A.pth'
WEIGHTS_SHA256 = 'bedb611875983a7db58a0b1ff65efeef1fb595dd6cfd0f53292015fb8a87a989'
WEIGHTS_SIZE_MB = 145

LICENSE_SETTINGS_KEY = 'MRtoCT/LicenseAccepted'
LICENSE_TEXT = (
    'MR to CT uses a deep-learning model distributed under the Creative Commons '
    'Attribution-NonCommercial 4.0 license (CC BY-NC 4.0):\n\n'
    ' - non-commercial use only;\n'
    ' - research use only: the synthetic CT is NOT validated for clinical use or treatment planning.\n\n'
    f'On first use the model (about {WEIGHTS_SIZE_MB} MB) is downloaded once.\n\n'
    'Do you accept these terms?'
)


class MRtoCT(ScriptedLoadableModule):
    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = 'MR to CT'
        self.parent.categories = ['Radiotherapy']
        self.parent.dependencies = []
        self.parent.contributors = ['Kevin Estiven Giraldo Paniagua']
        self.parent.helpText = (
            'Generates a synthetic CT (in HU) from a brain MR with the SAdaNGF generator. '
            'The MR is scaled to [-1, 1] (percentiles 1-99.9), resampled to 1 mm, translated with '
            '128^3 sliding windows and resampled back, so the sCT is on the same grid as the MR. '
            'Research use only, not for clinical use. Non-commercial license (CC BY-NC 4.0).'
        )
        self.parent.acknowledgementText = 'Licensed under CC BY-NC 4.0. Research use only.'


class MRtoCTWidget(ScriptedLoadableModuleWidget):
    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)
        self.logic = MRtoCTLogic()

        box = ctk.ctkCollapsibleButton()
        box.text = 'Synthetic CT'
        self.layout.addWidget(box)
        form = qt.QFormLayout(box)

        self.inputSelector = slicer.qMRMLNodeComboBox()
        self.inputSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.inputSelector.selectNodeUponCreation = True
        self.inputSelector.addEnabled = False
        self.inputSelector.removeEnabled = False
        self.inputSelector.noneEnabled = False
        self.inputSelector.showHidden = False
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.inputSelector.setToolTip('MR volume to translate.')
        form.addRow('Input MR:', self.inputSelector)

        self.outputSelector = slicer.qMRMLNodeComboBox()
        self.outputSelector.nodeTypes = ['vtkMRMLScalarVolumeNode']
        self.outputSelector.selectNodeUponCreation = True
        self.outputSelector.addEnabled = True
        self.outputSelector.renameEnabled = True
        self.outputSelector.removeEnabled = True
        self.outputSelector.noneEnabled = True
        self.outputSelector.noneDisplay = '(Create new volume)'
        self.outputSelector.showHidden = False
        self.outputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setToolTip('Volume that receives the synthetic CT (HU), on the MR grid.')
        form.addRow('Output sCT:', self.outputSelector)

        self.gpuCheckBox = qt.QCheckBox()
        self.gpuCheckBox.checked = True
        self.gpuCheckBox.setToolTip('Use the NVIDIA GPU if PyTorch can see it; otherwise the CPU is used (slow).')
        form.addRow('Use GPU:', self.gpuCheckBox)

        self.applyButton = qt.QPushButton('Generate synthetic CT')
        form.addRow(self.applyButton)
        self.statusLabel = qt.QLabel('')
        self.statusLabel.wordWrap = True
        form.addRow(self.statusLabel)

        advanced = ctk.ctkCollapsibleButton()
        advanced.text = 'Advanced'
        advanced.collapsed = True
        self.layout.addWidget(advanced)
        advancedForm = qt.QFormLayout(advanced)

        self.batchSpinBox = qt.QSpinBox()
        self.batchSpinBox.minimum = 1
        self.batchSpinBox.maximum = 16
        self.batchSpinBox.value = 1
        self.batchSpinBox.setToolTip(
            'Number of 128^3 patches per forward pass. Does not change the result; 1 fits an 8 GB GPU, '
            'raise it on bigger GPUs to go faster.'
        )
        advancedForm.addRow('Patches per batch:', self.batchSpinBox)

        self.weightsSelector = ctk.ctkPathLineEdit()
        self.weightsSelector.filters = ctk.ctkPathLineEdit.Files
        self.weightsSelector.nameFilters = ['PyTorch weights (*.pth *.pt)']
        self.weightsSelector.settingKey = 'MRtoCT/CustomWeights'
        self.weightsSelector.setToolTip('Leave empty to use the published model (downloaded on first use).')
        advancedForm.addRow('Custom weights:', self.weightsSelector)

        self.applyButton.connect('clicked(bool)', self.onApply)
        self.layout.addStretch(1)

    def onApply(self):
        inputVolume = self.inputSelector.currentNode()
        if inputVolume is None:
            slicer.util.errorDisplay('Select an input MR volume.')
            return
        if not MRtoCTLogic.licenseAccepted(askUser=True):
            return
        customWeights = self.weightsSelector.currentPath or None
        if customWeights:
            self.weightsSelector.addCurrentPathToHistory()

        outputVolume = self.outputSelector.currentNode()
        if outputVolume is None:
            name = slicer.mrmlScene.GenerateUniqueName(inputVolume.GetName() + '_sCT')
            outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', name)
            self.outputSelector.setCurrentNode(outputVolume)

        self.statusLabel.text = 'Running...'
        slicer.app.processEvents()
        try:
            with slicer.util.tryWithErrorDisplay('Synthetic CT generation failed.', waitCursor=True):
                device, seconds = self.logic.process(
                    inputVolume, outputVolume, customWeights, useGPU=self.gpuCheckBox.checked,
                    slidingWindow={'sw_batch_size': self.batchSpinBox.value})
                self.statusLabel.text = f'Done on {device} in {seconds:.1f} s.'
        finally:
            if self.statusLabel.text.startswith('Running'):
                self.statusLabel.text = ('Failed: see the error message. If the GPU ran out of memory, '
                                         'untick "Use GPU" or keep "Patches per batch" at 1.')


class MRtoCTLogic(ScriptedLoadableModuleLogic):
    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self._net = None
        self._netKey = None

    # ---- license -------------------------------------------------------------------------
    @staticmethod
    def licenseAccepted(askUser=True):
        settings = qt.QSettings()
        if str(settings.value(LICENSE_SETTINGS_KEY, '')).lower() == 'true':
            return True
        if not askUser or not slicer.util.confirmYesNoDisplay(LICENSE_TEXT, windowTitle='MR to CT: license'):
            return False
        settings.setValue(LICENSE_SETTINGS_KEY, 'true')
        return True

    # ---- dependencies ----------------------------------------------------------------------
    @staticmethod
    def ensureDependencies(askConfirmation=True):
        """Only PyTorch is needed. MONAI / torchvision are deliberately not used: importing them loads
        triton, whose native library can crash the Slicer application on Linux."""
        try:
            import torch  # noqa: F401
        except ImportError:
            try:
                import PyTorchUtils
            except ImportError:
                raise RuntimeError('PyTorch is missing: install the "PyTorch" extension from the Extensions Manager.')
            if PyTorchUtils.PyTorchUtilsLogic().torch is None:   # installs PyTorch (asks the user)
                raise RuntimeError('PyTorch installation was cancelled or failed (see the PyTorch Utils module).')

    # ---- weights ---------------------------------------------------------------------------
    @staticmethod
    def weightsCacheFolder():
        return os.path.join(slicer.app.cachePath, 'MRtoCT')

    @staticmethod
    def sha256(path):
        h = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(1 << 20), b''):
                h.update(chunk)
        return h.hexdigest()

    def resolveWeights(self, customWeights=None):
        """Custom file if given, else best_net_G_A.pth in the module folder, else the cached
        download, else download it now."""
        if customWeights:
            if not os.path.isfile(customWeights):
                raise FileNotFoundError(customWeights)
            return customWeights
        local = os.path.join(os.path.dirname(os.path.abspath(__file__)), WEIGHTS_FILENAME)
        if os.path.isfile(local):
            return local
        cached = os.path.join(self.weightsCacheFolder(), WEIGHTS_FILENAME)
        if not os.path.isfile(cached):
            self.downloadWeights(cached)
        return cached

    def downloadWeights(self, target):
        if 'kgiraldo09' in WEIGHTS_URL:
            raise RuntimeError('WEIGHTS_URL is not set in MRtoCT.py (see README, step 3).')
        os.makedirs(os.path.dirname(target), exist_ok=True)
        partial = target + '.part'
        progress = slicer.util.createProgressDialog(
            labelText=f'Downloading the MR to CT model (about {WEIGHTS_SIZE_MB} MB)...', maximum=100)

        def report(blocks, blockSize, total):
            if total > 0:
                progress.value = min(100, int(blocks * blockSize * 100 / total))
            slicer.app.processEvents()
            if progress.wasCanceled:
                raise RuntimeError('Download cancelled.')

        try:
            logging.info(f'MRtoCT: downloading {WEIGHTS_URL}')
            urllib.request.urlretrieve(WEIGHTS_URL, partial, report)
        except Exception:
            if os.path.exists(partial):
                os.remove(partial)
            raise
        finally:
            progress.close()
        if WEIGHTS_SHA256 and self.sha256(partial) != WEIGHTS_SHA256.lower():
            os.remove(partial)
            raise RuntimeError('The downloaded model is corrupted (checksum mismatch). Please try again.')
        os.replace(partial, target)

    def loadNetwork(self, weights, device, ngf=64):
        from MRtoCTLib import mr_to_sct

        key = (str(weights), os.path.getmtime(weights), str(device), ngf)
        if self._netKey != key:
            self._net = mr_to_sct.load_generator(weights, device, ngf=ngf)
            self._netKey = key
        return self._net

    # ---- processing ------------------------------------------------------------------------
    def process(self, inputVolume, outputVolume, customWeights=None, useGPU=True, slidingWindow=None,
                showResult=True, ngf=64):
        """Returns (device name, seconds)."""
        start = time.time()
        self.ensureDependencies()
        import torch

        from MRtoCTLib import mr_to_sct

        weights = self.resolveWeights(customWeights)
        device = torch.device('cuda' if useGPU and torch.cuda.is_available() else 'cpu')
        if useGPU and device.type == 'cpu':
            logging.warning('MRtoCT: PyTorch cannot see a GPU, running on the CPU (slow).')
        net = self.loadNetwork(weights, device, ngf)

        # Slicer arrays are (k, j, i); mr_to_sct works in NIfTI voxel order (i, j, k)
        mr_ijk = slicer.util.arrayFromVolume(inputVolume).transpose(2, 1, 0)
        try:
            sct_ijk = mr_to_sct.mr_to_sct(mr_ijk, inputVolume.GetSpacing(), net, device, sliding_window_params=slidingWindow)
        finally:
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        sct = np.ascontiguousarray(sct_ijk.transpose(2, 1, 0))

        # same voxel grid and position as the input MR
        ijkToRas = vtk.vtkMatrix4x4()
        inputVolume.GetIJKToRASMatrix(ijkToRas)
        outputVolume.SetIJKToRASMatrix(ijkToRas)
        slicer.util.updateVolumeFromArray(outputVolume, sct)
        outputVolume.SetAndObserveTransformNodeID(inputVolume.GetTransformNodeID())
        outputVolume.CreateDefaultDisplayNodes()
        displayNode = outputVolume.GetDisplayNode()
        displayNode.SetAutoWindowLevel(False)
        displayNode.SetWindowLevelMinMax(-1000, 1500)

        if showResult:
            slicer.util.setSliceViewerLayers(background=outputVolume, foreground=None, fit=True)
        return device.type.upper(), time.time() - start


class MRtoCTTest(ScriptedLoadableModuleTest):
    """Checks the Slicer plumbing with a small random generator (no download needed): the module's
    sCT must equal what mr_to_sct.py gives on the same array, with the MR's geometry."""

    def runTest(self):
        slicer.mrmlScene.Clear()
        self.test_matches_mr_to_sct()

    def test_matches_mr_to_sct(self):
        import tempfile
        from pathlib import Path

        import torch

        self.delayDisplay('Module output vs mr_to_sct.py')
        MRtoCTLogic.ensureDependencies(askConfirmation=False)
        from MRtoCTLib import mr_to_sct

        torch.manual_seed(0)
        weights = Path(tempfile.mkdtemp()) / 'tiny.pth'
        torch.save(mr_to_sct.ResnetGenerator3(ngf=4).state_dict(), weights)
        small = {'roi_size': (16, 16, 16), 'overlap': 0.5, 'sw_batch_size': 2}

        rng = np.random.default_rng(0)
        mr_kji = rng.gamma(2.0, 150.0, size=(18, 26, 31)).astype(np.float32)
        inputVolume = slicer.util.addVolumeFromArray(mr_kji, name='mr')
        inputVolume.SetSpacing(0.9, 1.2, 1.6)
        inputVolume.SetOrigin(10, -20, 30)
        outputVolume = slicer.mrmlScene.AddNewNodeByClass('vtkMRMLScalarVolumeNode', 'sct')

        MRtoCTLogic().process(inputVolume, outputVolume, str(weights), useGPU=False, slidingWindow=small,
                              showResult=False, ngf=4)

        net = mr_to_sct.load_generator(weights, 'cpu', ngf=4)
        expected = mr_to_sct.mr_to_sct(mr_kji.transpose(2, 1, 0), (0.9, 1.2, 1.6), net, 'cpu',
                                       sliding_window_params=small).transpose(2, 1, 0)
        sct = slicer.util.arrayFromVolume(outputVolume)
        self.assertEqual(sct.shape, mr_kji.shape)
        np.testing.assert_allclose(sct, expected, atol=1e-3)
        np.testing.assert_allclose(outputVolume.GetSpacing(), inputVolume.GetSpacing())
        np.testing.assert_allclose(outputVolume.GetOrigin(), inputVolume.GetOrigin())
        self.delayDisplay('Test passed')
