import { app } from "/scripts/app.js";

const PACK_ID = "comfyui-cyberdelia-metadata";
const PACK_VERSION = "2.0.0";
const LEGACY_TO_CANONICAL = Object.freeze({
  SaveImageWithMetaData: "CyberdeliaSaveImageWithMetaData",
  CreateExtraMetaData: "CyberdeliaCreateExtraMetaData",
});

/**
 * Upgrade serialized nodes that are known to belong to this pack.
 *
 * The legacy ids are also valid ids from nkchocoai's original pack, so a
 * global replacement would hijack unrelated workflows. `cnr_id` makes this
 * migration pack-specific and lets both packs coexist.
 */
export function migrateCyberdeliaWorkflow(graphData) {
  const seen = new WeakSet();
  let migrated = 0;

  const visit = (value) => {
    if (value === null || typeof value !== "object" || seen.has(value)) {
      return;
    }
    seen.add(value);

    if (!Array.isArray(value)) {
      const properties = value.properties;
      const replacement = LEGACY_TO_CANONICAL[value.type];
      if (
        replacement &&
        properties &&
        typeof properties === "object" &&
        (properties.cnr_id === PACK_ID || properties.aux_id === PACK_ID)
      ) {
        const legacyType = value.type;
        value.type = replacement;
        properties.cnr_id = PACK_ID;
        properties.ver = PACK_VERSION;
        properties["Node name for S&R"] = replacement;
        migrated += 1;
        console.info(
          `[Cyberdelia Metadata] Migrated ${legacyType} to ${replacement}.`,
        );
      }
    }

    for (const nested of Object.values(value)) {
      visit(nested);
    }
  };

  visit(graphData);
  return migrated;
}

app.registerExtension({
  name: "Cyberdelia.Metadata.WorkflowMigration",
  beforeConfigureGraph(graphData) {
    migrateCyberdeliaWorkflow(graphData);
  },
});
