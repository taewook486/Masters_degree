# docx -> PDF 변환 (Word COM, 후기 바인딩).
#
# 학위논문 제출본은 PDF이므로, 양식(B5/여백/장평)이 살아 있는 docx를
# Word로 열어 목차 필드를 갱신한 뒤 PDF로 내보낸다.
#
# 초기 바인딩(New-Object 후 직접 속성 접근)은 이 환경에서
# TYPE_E_CANTLOADLIBRARY로 실패하므로 InvokeMember 방식을 쓴다.
#
# 사용법:
#   powershell -File scripts/word_to_pdf.ps1 -InPath D:\...\a.docx -OutPath D:\...\a.pdf
param(
    [Parameter(Mandatory = $true)][string]$InPath,
    [Parameter(Mandatory = $true)][string]$OutPath
)

$ErrorActionPreference = 'Stop'

# PowerShell은 함수가 돌려주는 열거 가능 객체를 자동으로 펼친다.
# COM 컬렉션(Documents 등)이 비어 있으면 그 결과가 $null이 되어버리므로
# -NoEnumerate로 펼침을 막아야 한다.
function Get-Prop($obj, $name, $argv = @()) {
    # 인자를 $null로 넘기면 "null 값 식에서 메서드를 호출할 수 없습니다"로 실패한다.
    $r = $obj.GetType().InvokeMember($name, 'GetProperty', $null, $obj, $argv)
    Write-Output -NoEnumerate $r
}
function Set-Prop($obj, $name, $value) {
    $obj.GetType().InvokeMember($name, 'SetProperty', $null, $obj, @($value)) | Out-Null
}
function Invoke-Member2($obj, $name, $argv = @()) {
    $r = $obj.GetType().InvokeMember($name, 'InvokeMethod', $null, $obj, $argv)
    Write-Output -NoEnumerate $r
}

$word = $null
$doc = $null
try {
    $word = New-Object -ComObject Word.Application
    Write-Output 'STEP: app created'
    Set-Prop $word 'Visible' $false
    Set-Prop $word 'DisplayAlerts' 0
    Write-Output 'STEP: props set'

    $docs = Get-Prop $word 'Documents'
    Write-Output 'STEP: documents got'
    # Open(FileName, ConfirmConversions, ReadOnly)
    $doc = Invoke-Member2 $docs 'Open' @($InPath, $false, $false)
    Write-Output ('STEP: opened, null? ' + ($null -eq $doc))

    # 목차 필드 갱신. 페이지 번호가 확정되려면 갱신-재페이지네이션을 두 번 돈다.
    $tocs = Get-Prop $doc 'TablesOfContents'
    $tocCount = Get-Prop $tocs 'Count'
    for ($pass = 1; $pass -le 2; $pass++) {
        for ($i = 1; $i -le $tocCount; $i++) {
            $toc = Invoke-Member2 $tocs 'Item' @($i)
            Invoke-Member2 $toc 'Update' @() | Out-Null
        }
        $fields = Get-Prop $doc 'Fields'
        Invoke-Member2 $fields 'Update' @() | Out-Null
        Invoke-Member2 $doc 'Repaginate' @() | Out-Null
    }

    # ComputeStatistics(2) = wdStatisticPages
    $pages = Invoke-Member2 $doc 'ComputeStatistics' @(2)
    # ExportAsFixedFormat(OutputFileName, ExportFormat) / 17 = wdExportFormatPDF
    Invoke-Member2 $doc 'ExportAsFixedFormat' @($OutPath, 17) | Out-Null

    Write-Output ("TOC_COUNT=" + $tocCount)
    Write-Output ("PAGES=" + $pages)
    Write-Output "OK"
}
catch {
    Write-Output ("FAIL: " + $_.Exception.Message)
}
finally {
    if ($null -ne $doc) { try { Invoke-Member2 $doc 'Close' @(0) | Out-Null } catch {} }
    if ($null -ne $word) { try { Invoke-Member2 $word 'Quit' @() | Out-Null } catch {} }
}
